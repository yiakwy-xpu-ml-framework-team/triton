"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)

Sequence Parallel implementation inspired by HazyResearch
(see https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
"""

import torch

from .. import cdiv, jit
from .. import language as tl

from enum import Enum

@jit
def _fwd_kernel_compute(acc, m_i, l_i, q, vDtype, K_block_ptr, V_block_ptr, start_m,
                        offs_m, offs_n, BLOCK_M, BLOCK_N,
                        IS_CAUSAL, N_CTX):
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q.dtype)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(vDtype), v, allow_tf32=True)
        
        # -- update m_i and l_i --        
        l_i *= alpha
        l_i += tl.sum(p, 1)
        m_i = m_i_new
        
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    return acc, l_i, m_i

@jit
def _fwd_kernel_without_tma(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_DIM_0: tl.constexpr,
    LOAD_BALANCE_STRATEGY : tl.constexpr
):
    start_m = tl.program_id(0) # will be updated for causual attention
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # will be updated for causual attention
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(K.dtype.element_ty)
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # initialize offsets
    rows_q = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + rows_q # will be updated for causual attention
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=q.dtype) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Compute online softmax and memory efficient attentin algorithms with
    # tl.program_id(0) blocks along K, V dimensions iterately
    acc, l_i, m_i = _fwd_kernel_compute(acc, m_i, l_i,
                                        q, V.dtype.element_ty, K_block_ptr, V_block_ptr, start_m,
                                        offs_m, offs_n, BLOCK_M, BLOCK_N,
                                        IS_CAUSAL, N_CTX)
        
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    
    # write back O with tl.program_id(0) * BLOCK_M
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))
    
    if not IS_CAUSAL or LOAD_BALANCE_STRATEGY < 0:
        return

    if BLOCK_DIM_0 - start_m == start_m:
        return

    # Do load balance
    if LOAD_BALANCE_STRATEGY == 0:
        # gaussian loading
        start_m = BLOCK_DIM_0 - start_m
    
    # Load new Q, K, V for computation balancing and resuse SRAM allocated    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(K.dtype.element_ty)
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    offs_m = start_m * BLOCK_M + rows_q
    
    # Reinitalize SRAM buffer for online softmax and memory efficient attentin algorithms
    m_i = tl.where(rows_q, m_i, -float("inf"))
    # this can saves half of acc memroy when BLOCK_DMODEL increases to 128 or 256
    l_i -= l_i
    acc -= acc 

    # Compute online softmax and memory efficient attentin algorithms with 
    # (sequence_len - tl.program_id(0)) blocks along K, V dimensions iterately
    acc, l_i, m_i = _fwd_kernel_compute(acc, m_i, l_i,
                                        q, V.dtype.element_ty, K_block_ptr, V_block_ptr, start_m,
                                        offs_m, offs_n, BLOCK_M, BLOCK_N,
                                        IS_CAUSAL, N_CTX)
        
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    
    # write back O with (offset BLOCK_DIM_0: - tl.program_id(0)) * BLOCK_M
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))

@jit
def _fwd_kernel(Q, K, V, sm_scale,  #
                L,  #
                Out,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                IS_CAUSAL: tl.constexpr,  #
                BLOCK_DIM_0: tl.constexpr, #
                LOAD_BALANCE_STRATEGY : tl.constexpr #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
        
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, Z_H_N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, vk_offset),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(vk_offset, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout

    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs)

    q = (q * qk_scale).to(K.dtype.element_ty)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(vk_offset + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


@jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    # (BLOCK_M, EMB_D), row major vectors
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@jit
def _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale,  #
                              Out, DO,  #
                              DQ, DK, DV,  #
                              L,  #
                              D,  #
                              Q_block_ptr, K_block_ptr, V_block_ptr,  #
                              DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                              stride_dqa, #
                              stride_qz, stride_qh, stride_qm, stride_qk,  #
                              stride_kz, stride_kh, stride_kn, stride_kk,  #
                              stride_vz, stride_vh, stride_vn, stride_vk,  #
                              Z, H, N_CTX,  #
                              off_h, off_z, off_hz, start_n, num_block_n,  #
                              BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                              BLOCK_N: tl.constexpr,  #
                              SEQUENCE_PARALLEL: tl.constexpr,  #
                              IS_CAUSAL: tl.constexpr,  #
                              MMA_V3: tl.constexpr  #
                              ):
    lo = (start_n * BLOCK_M) % N_CTX if IS_CAUSAL else 0
    
    Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
    DQ_offset = off_z * stride_qz + off_h * stride_qh
    K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
    V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
    if SEQUENCE_PARALLEL:
        DQ_offset += stride_dqa * start_n
    DQ_offset = DQ_offset // stride_qm

    Q_block_ptr = tl.advance(Q_block_ptr, (lo + Q_offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_N + K_offset, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_N + V_offset, 0))
    
    DO_block_ptr = tl.advance(DO_block_ptr, (lo + Q_offset, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
    
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_N + K_offset, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_N + V_offset, 0))

    # initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    # initialize dv amd dk
    # NOTE dv is an accumulator, see analysis below, hence FP32 is selected
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # NOTE Note dk is an accumulator, see analysis below, hence FP32 is selected
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # k and v stay in SRAM throughout
    # load (BLOCK_SIZE_N, EMB_D), at start_n (jth) columns (rowswise if K is not transposed) of K
    k = tl.load(K_block_ptr)
    # load (BLOCK_SIZE_N, EMB_D), at start_n (jth) rows of V
    v = tl.load(V_block_ptr)
            
    # loop over rows
    hi = N_CTX
    for start_m in range(lo, hi, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # load (BLOCK_SIZE_M, EMB_D), at start_m (ith) rows of Q
        q = tl.load(Q_block_ptr)
        l_i = tl.load(l_ptrs + offs_m_curr)
        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        
        # -- Recompute Stage Begin --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q.dtype)
        # TODO (yiakwy) : causal mask means qk_ij (j>=i) has no contribution to output
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, tl.trans(k), allow_tf32=True)
        qk *= qk_scale
        p = tl.math.exp2(qk - l_i[:, None])
        
        # TODO (yiakwy) : missing dropout
        
        # -- Recompute Stage End -- 
        
        # compute dv
        do = tl.load(DO_block_ptr)
        #   [    mask   ]       [ do_1 ]
        # ( [...p_2_j...] ).T x [ do_2 ] = dv, reduced to summation form
        #   [...p_3_j...]       [ do_3 ]
        # dv is an accumulator with tile of p and tile of do
        # TDOO (yiakwy) : decide which p and do contribute to dv under causal mask
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        
        # compute dp
        # [ do_1 ]
        # [ do_2 ] x [ v_1.T v_2.T v_3.T ] = ( dp_ij ), dp_ij = do_i x v_j.T
        # [ do_3 ]
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        
        # compute ds [BLOCK_SIZE_M, BLOCK_SIZE_N]
        Di = tl.load(D_ptrs + offs_m_curr)
        # ds_ij = p_ij .* (dp_ij - Di[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        
        # compute dk
        #   [...ds_1_j...]       [ q_1 ]
        # ( [...ds_2_j...] ).T x [ q_2 ] = dk, reduced to summation form
        #   [...ds_3_j...]       [ q_3 ]
        # dk is an accumulator with tile of ds_ij and tile of q_i
        # TDOO (yiakwy) : decide which ds and q contribute to dk under causal mask
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True)
        
        # compute dq
        #                            [ k_0 ]
        #   [ds_i_1 ds_i_2 ds_i_3] x [ k_1 ] = sum ( ds_ij * k_j ) (Note K is not transposed here)
        #                            [ k_2 ]
        # hence dq is an global accumulator of ds_ij * k_j for different column j
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k, allow_tf32=True)
            if dq.dtype != Q.dtype.element_ty:
                dq = dq.to(Q.dtype.element_ty)
            tl.store(DQ_block_ptr, dq)
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k, allow_tf32=True)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds), allow_tf32=True))
            if dq.dtype != Q.dtype.element_ty:
                dq = dq.to(Q.dtype.element_ty)
            tl.store(DQ_block_ptr, dq)

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        
    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))


@jit
def _bwd_kernel_one_col_block_for_dv_dk(Q, K, V, sm_scale, qk_scale,  #
                                        Out, DO,  #
                                        DQ, DK, DV,  #
                                        L,  #
                                        D,  #
                                        Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                        DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                        stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                        stride_kz, stride_kh, stride_kn, stride_kk,  #
                                        stride_vz, stride_vh, stride_vn, stride_vk,  #
                                        Z, H, N_CTX,  #
                                        off_h, off_z, off_hz, start_n, num_block_n,  #
                                        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                                        BLOCK_N: tl.constexpr,  #
                                        SEQUENCE_PARALLEL: tl.constexpr,  #
                                        IS_CAUSAL: tl.constexpr,  #
                                        BLOCK_DIM_1: tl.constexpr,  #
                                        STORE_DQ: tl.constexpr,  # 
                                        MMA_V3: tl.constexpr  #
                                        ):
    # NOTE(yiakwy) if BLOCK_M != BLOCK_N, the load blance changed
    lo = (start_n * BLOCK_M) % N_CTX if IS_CAUSAL else 0
    
    # advanced pointers
    if STORE_DQ:
        DQ_offset = ((BLOCK_DIM_1 - start_n) * stride_dqa + off_z * stride_qz + off_h * stride_qh) // stride_qm
    else:
        DQ_offset = (start_n * stride_dqa + off_z * stride_qz + off_h * stride_qh) // stride_qm
      
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
    Q_block_ptr = tl.advance(Q_block_ptr,   (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))

    # initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
        
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    
    # initialize dv amd dk
    # NOTE dv is an accumulator, see analysis below, hence FP32 is selected
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # NOTE dk is an accumulator, see analysis below, hence FP32 is selected
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # k and v stay in SRAM throughout
    # load (BLOCK_SIZE_N, EMB_D), at start_n (jth) columns (rowswise if K is not transposed) of K
    k = tl.load(K_block_ptr)
    # load (BLOCK_SIZE_N, EMB_D), at start_n (jth) rows of V
    v = tl.load(V_block_ptr)
        
    # loop over rows
    hi = N_CTX
    for start_m in range(lo, hi, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # load (BLOCK_SIZE_M, EMB_D), at start_m (ith) rows of Q
        q = tl.load(Q_block_ptr)
        l_i = tl.load(l_ptrs + offs_m_curr)
        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        
        # -- Recompute Stage Begin --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q.dtype)
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, tl.trans(k), allow_tf32=True)
        qk *= qk_scale
        p = tl.math.exp2(qk - l_i[:, None])
                        
        # TODO (yiakwy) : missing dropout
        
        # -- Recompute Stage End -- 
        
        # compute dv
        do = tl.load(DO_block_ptr)
        #   [    mask   ]       [ do_1 ]
        # ( [...p_2_j...] ).T x [ do_2 ] = dv, reduced to summation form
        #   [...p_3_j...]       [ do_3 ]
        # dv is an accumulator with tile of p and tile of do
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        
        # compute dp
        # [ do_1 ]
        # [ do_2 ] x [ v_1.T v_2.T v_3.T ] = ( dp_ij ), dp_ij = do_i x v_j.T
        # [ do_3 ]
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        
        # compute ds [BLOCK_SIZE_M, BLOCK_SIZE_N]
        Di = tl.load(D_ptrs + offs_m_curr)
        # ds_ij = p_ij .* (dp_ij - Di[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        
        # compute dk
        #   [...ds_1_j...]       [ q_1 ]
        # ( [...ds_2_j...] ).T x [ q_2 ] = dk, reduced to summation form
        #   [...ds_3_j...]       [ q_3 ]
        # dk is an accumulator with tile of ds_ij and tile of q_i
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True)
        
        # compute dq
        #                            [ k_0 ]
        #   [ds_i_1 ds_i_2 ds_i_3] x [ k_1 ] = sum ( ds_ij * k_j ) (Note K is not transposed here)
        #                            [ k_2 ]
        # hence dq is an global accumulator of ds_ij * k_j for different column j
        # TODO (yiakwy) : balance op store to facsilitate load/store/compute overlap
        # TODO (yiakwy) : decide which ds and k contribute to dq under causal mask
        if MMA_V3:
            if STORE_DQ:
                dq = tl.load(DQ_block_ptr)
                dq += tl.dot(ds, k, allow_tf32=True)
            else:
                dq = tl.dot(ds, k, allow_tf32=True)
        else:
            if STORE_DQ:
                dq = tl.load(DQ_block_ptr)
                dq += tl.dot(ds, k, allow_tf32=True)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds), allow_tf32=True))
        # TODO (yiakwy) : reduce store
        if dq.dtype != Q.dtype.element_ty:
            dq = dq.to(Q.dtype.element_ty)
        # store the partial dq on the replica dimension
        tl.store(DQ_block_ptr, dq)

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        
    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))

@jit
def _bwd_kernel(Q, K, V, sm_scale,  #
                Out, DO,  #
                DQ, DK, DV,  #
                L,  #
                D,  #
                stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                SQ_Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                SEQUENCE_PARALLEL: tl.constexpr,  #
                IS_CAUSAL: tl.constexpr,  #
                MMA_V3: tl.constexpr  #
                ):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if SEQUENCE_PARALLEL:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(SQ_Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    else:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO,  #
                                      DQ, DK, DV,  #
                                      L,  #
                                      D,  #
                                      Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                      DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                      stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                      stride_kz, stride_kh, stride_kn, stride_kk,  #
                                      stride_vz, stride_vh, stride_vn, stride_vk,  #
                                      Z, H, N_CTX,  #
                                      off_h, off_z, off_hz, start_n, num_block_n,  #
                                      BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                      BLOCK_N=BLOCK_N,  #
                                      SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                      IS_CAUSAL=IS_CAUSAL,  #
                                      MMA_V3=MMA_V3  #
                                      )
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO,  #
                                  DQ, DK, DV,  #
                                  L,  #
                                  D,  #
                                  Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                  DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                  stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                  stride_kz, stride_kh, stride_kn, stride_kk,  #
                                  stride_vz, stride_vh, stride_vn, stride_vk,  #
                                  Z, H, N_CTX,  #
                                  off_h, off_z, off_hz, start_n, num_block_n,  #
                                  BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                  BLOCK_N=BLOCK_N,  #
                                  SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                  IS_CAUSAL=IS_CAUSAL,  #
                                  MMA_V3=MMA_V3  #
                                  )

@jit
def _bwd_kernel_without_tma(Q, K, V, sm_scale,  #
                            Out, DO,  #
                            DQ, DK, DV,  #
                            L,  #
                            D,  #
                            stride_dqa, #
                            stride_qz, stride_qh, stride_qm, stride_qk,  #
                            stride_kz, stride_kh, stride_kn, stride_kk,  #
                            stride_vz, stride_vh, stride_vn, stride_vk,  #
                            Z, H, N_CTX,  #
                            Z_H_N_CTX,  #
                            SQ_Z_H_N_CTX,  #
                            BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                            BLOCK_N: tl.constexpr,  #
                            SEQUENCE_PARALLEL: tl.constexpr,  #
                            IS_CAUSAL: tl.constexpr,  #
                            BLOCK_DIM_1: tl.constexpr,
                            LOAD_BALANCE_STRATEGY: tl.constexpr,  #
                            MMA_V3: tl.constexpr  #
                            ):
    off_hz = tl.program_id(0)
    start_n = tl.program_id(1)
    
    qk_scale = sm_scale * 1.44269504
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_hz * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    # NOTE DQ is of shape ( N_CTX / BLOCK_N, Z, H, N_CTX, ) where first dim N_CTX / BLOCK_N is replicated
    # to store dq for each column partition
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)

    _bwd_kernel_one_col_block_for_dv_dk(Q, K, V, sm_scale, qk_scale, Out, DO,  #
                                        DQ, DK, DV,  #
                                        L,  #
                                        D,  #
                                        Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                        DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                        stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                        stride_kz, stride_kh, stride_kn, stride_kk,  #
                                        stride_vz, stride_vh, stride_vn, stride_vk,  #
                                        Z, H, N_CTX,  #
                                        off_h, off_z, off_hz, start_n, num_block_n,  #
                                        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                        BLOCK_N=BLOCK_N,  #
                                        SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                        IS_CAUSAL=IS_CAUSAL,  #
                                        BLOCK_DIM_1=BLOCK_DIM_1,  #
                                        STORE_DQ=False, 
                                        MMA_V3=MMA_V3  #,
                                        )
    
    # load balance
    if not IS_CAUSAL or LOAD_BALANCE_STRATEGY < 0:
        return

    if BLOCK_DIM_1 - start_n == start_n:
        return

    # Do load balance
    if LOAD_BALANCE_STRATEGY == 0:
        # gaussian loading
        start_n = BLOCK_DIM_1 - start_n
    
    # reload data
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    _bwd_kernel_one_col_block_for_dv_dk(Q, K, V, sm_scale, qk_scale, Out, DO,  #
                                        DQ, DK, DV,  #
                                        L,  #
                                        D,  #
                                        Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                        DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                        stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                        stride_kz, stride_kh, stride_kn, stride_kk,  #
                                        stride_vz, stride_vh, stride_vn, stride_vk,  #
                                        Z, H, N_CTX,  #
                                        off_h, off_z, off_hz, start_n, num_block_n,  #
                                        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                        BLOCK_N=BLOCK_N,  #
                                        SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                        IS_CAUSAL=IS_CAUSAL,  #
                                        BLOCK_DIM_1=BLOCK_DIM_1,  #
                                        STORE_DQ=True,  #
                                        MMA_V3=MMA_V3  #
                                        )
    pass


class _attention(torch.autograd.Function):

    class LoadBalanceStrategy(Enum):
        UNDEFINE = -1
        GAUSSIAN_FOLDING = 0
        pass

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, sequence_parallel=False, load_balance_strategy=0):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_M = 128
        BLOCK_N = 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        
        o = torch.empty_like(q)
        
        block_dim_0 = cdiv(q.shape[2], BLOCK_M)
        if causal:
            if load_balance_strategy == _attention.LoadBalanceStrategy.GAUSSIAN_FOLDING.value:
                # make sure 
                if q.shape[2] % 2 != 0:
                    raise Exception("Odd Lq is not handled in guassian folding algo.")
                
                # treat the data of shape (Z(B), H, sequence / 2, emb_d)
                if q.shape[2] < 2*BLOCK_M:
                    BLOCK_M = cdiv(q.shape[2], 2)
                    
                block_dim_0 = cdiv(cdiv(q.shape[2], 2), BLOCK_M)
            else:
                raise Exception(f"Unexpected load balance strategy; valid choices : {list(_attention.LoadBalanceStrategy)}")
        else:
            if load_balance_strategy > 0:
                print("load balance ignore for non-causal attention")
            
        
        grid = (block_dim_0, q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        if capability[0] >= 9:
            _fwd_kernel[grid](
                q, k, v, sm_scale,  #
                L,  #
                o,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1], q.shape[2],  #
                q.shape[0] * q.shape[1] * q.shape[2],  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,  #
                IS_CAUSAL=causal,  #
                LOAD_BALANCE_STRATEGY=load_balance_strategy,  #
                BLOCK_DIM_0=2*block_dim_0 - 1 if causal else block_dim_0,
                # TODO (yiakwy) : finetune
                num_warps=num_warps,  #
                # TODO (yiakwy) : finetune
                num_stages=4  #
            )
        else:
            # card with capability of 8.0 (A100, A800, A6000 ...) does not have TMA to acclerate memory loading
            # see PR#2336 https://github.com/openai/triton/pull/2336
            _fwd_kernel_without_tma[grid](
                q, k, v, sm_scale,
                L,
                o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
                IS_CAUSAL=causal,
                LOAD_BALANCE_STRATEGY=load_balance_strategy,
                BLOCK_DIM_0=2*block_dim_0 - 1 if causal else block_dim_0,
                # TODO (yiakwy) : finetune
                num_warps=num_warps,
                # TODO (yiakwy) : finetune
                num_stages=4
            )

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        ctx.load_balance_strategy = load_balance_strategy
        return o

    @staticmethod
    def backward(ctx, do):
        # NOTE uncomment the following lines to enable debugging triton program tile mapping
        # import pdb
        # pdb.set_trace()
        capability = torch.cuda.get_device_capability()
        MMA_V3 = capability[0] >= 9
        BLOCK_M = 128
        # TODO (yiakwy) The current impl does not support BLOCK_N != BLOCK_M
        BLOCK_N = BLOCK_M
        q, k, v, o, L = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        seq_len_kv = k.shape[2]
        do = do.contiguous()

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # (q.shape[0] * q.shape[1], q.shape[2]), a rowwise scalar used to compute row(S, i)
        # d row(S,i) = row(P, i) .* (d row(P, i) - delta(i))
        delta = torch.empty_like(L)
        _bwd_preprocess[(cdiv(q.shape[2], BLOCK_M) * ctx.grid[1],)](
            o,
            do,
            delta,
            BLOCK_M=BLOCK_M,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        
        # distribute blocks over SMs along batchesxheads x (seq_len_kv / BLOCK_SIZE_N)
        blocks_dim_1 = cdiv(seq_len_kv, BLOCK_N)
        reduction_dim = 0
        if capability[0] >= 9 or not sequence_parallel:
            if sequence_parallel:
                replicas = cdiv(seq_len_kv, BLOCK_N)
                new_dq_shape = (replicas, ) + q.shape
                dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
            else:
                dq = torch.zeros_like(q, dtype=q.dtype)
            grid = (ctx.grid[1], blocks_dim_1 if sequence_parallel else 1)
            _bwd_kernel[grid](
                q, k, v, ctx.sm_scale,  #
                o, do,  #
                dq, dk, dv,  #
                L,  #
                delta,  #
                o.numel(), #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                q.shape[0], q.shape[1], q.shape[2],  #
                q.shape[0] * q.shape[1] * q.shape[2],  #
                blocks_dim_1 * q.shape[0] * q.shape[1] * q.shape[2],  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
                SEQUENCE_PARALLEL=sequence_parallel,  #
                IS_CAUSAL=ctx.causal,  #
                MMA_V3=MMA_V3,  #
                num_warps=8,  #
                num_stages=1  #
            )
        else:
            # TODO (yiaky) : remove this constrains
            replicas = cdiv(cdiv(seq_len_kv, 2), BLOCK_N)
            # this is reduction friendly layout
            new_dq_shape = (replicas,) + q.shape
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
            reduction_dim = 0
            if ctx.causal:
                if ctx.load_balance_strategy == _attention.LoadBalanceStrategy.GAUSSIAN_FOLDING.value:
                    # make sure 
                    if seq_len_kv % 2 != 0:
                        raise Exception("Odd Lq is not handled in guassian folding algo.")
                    
                    # treat the data of shape (Z(B), H, sequence / 2, emb_d)
                    if seq_len_kv < 2*BLOCK_N:
                        BLOCK_N = cdiv(seq_len_kv, 2)
                        
                    blocks_dim_1 = cdiv(cdiv(seq_len_kv, 2), BLOCK_N)
                else:
                    raise Exception(f"Unexpected load balance strategy; valid choices : {list(_attention.LoadBalanceStrategy)}")
            else:
                if ctx.load_balance_strategy > 0:
                    print("load balance ignore for non-causal attention")
            
            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            
            # start.record()
            grid = (ctx.grid[1], blocks_dim_1)
            _bwd_kernel_without_tma[grid](
                q, k, v, ctx.sm_scale,  #
                o, do,  #
                dq, dk, dv,  #
                L,  #
                delta,  #
                o.numel(), #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                q.shape[0], q.shape[1], q.shape[2],  #
                q.shape[0] * q.shape[1] * q.shape[2],  #
                blocks_dim_1 * q.shape[0] * q.shape[1] * q.shape[2],  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
                SEQUENCE_PARALLEL=True,  #
                IS_CAUSAL=ctx.causal,  #
                BLOCK_DIM_1=2*blocks_dim_1 - 1 if ctx.causal else blocks_dim_1,
                LOAD_BALANCE_STRATEGY=ctx.load_balance_strategy,  #
                MMA_V3=MMA_V3,  #
                num_warps=8,  #
                num_stages=1  #
            )
            # end.record()
            
            # torch.cuda.synchronize()
            # print(f"[backward::_bwd_kernel_without_tma] {start.elapsed_time(end)} ms")

        # TODO (yiakwy) : Reduction could be extremely expensive for some layout
        # NOTE Flash Decoding has the same problem for fwd pass
        if len(dq.shape) == 5:
            # torch.cuda.synchronize()
            # start.record()
            dq = dq.sum(dim=reduction_dim)
            # end.record()
            
            # torch.cuda.synchronize()
            # print(f"[backward::reduce] {start.elapsed_time(end)} ms")
            
        return dq, dk, dv, None, None, None


attention = _attention.apply
