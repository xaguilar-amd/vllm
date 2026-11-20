# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B007, B023, E501, SIM102
# Copyright (c) 2025 FlyDSL Project Contributors

"""ML4 indexed decode MonoKernel: one persistent launch per TP rank.

One launch of `grid = 256 CTAs x 512 threads` (one CTA per MI355X CU) runs the
whole decoder-layer body for this rank's TP shard.  The indexed path adds the
selection refresh to the same resident grid instead of consuming externally
prepared sparse indices::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> index K/Q/W projection -> index K norm/RoPE/cache
      -> index score -> exact radix top-2048
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + activation FP8 quant
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> mid FP8 quant -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                   (sym_ffn)

Scheduling: every stage is a list of tasks; task `t` of a stage runs on CTA
`(stage_base + t) % 256` and every CTA walks the stages in order.  There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every spin wait makes progress.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, `(value, tag)`, with
device- (`sc1`) or system-coherent (`sc0 sc1`) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: weights are host-packed (`pack_fp8` /
`pack_bf16`) so one wave loads 16 rows x 64 k as one contiguous 1 KB, FP8 is
widened exactly to bf16 and fed to a 16x16x32 bf16 MFMA (`mfma_bf16`) with the samples
as the N dimension.  Each 64-k chunk's partial is scaled by its f32 block
scale (times any activation scale / route weight) into the accumulator, so the
math is exact block-scaled FP8 on bf16 activations.  Weight loads that do not
depend on upstream results are issued before the task waits for its inputs.

Cross-GPU: each rank pushes its partial rows as tagged pairs into every peer's
symmetric buffer and polls its own; every rank sums the TP partials in rank
order, so all ranks produce bit-identical hidden states (and routing).

TileRT shared/reuse MonoKernel reference (fusion-boundary comparison):
https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.arch import (
    GFX942,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    bpermute_i32,
    div_rn,
    f8_word,
    fp8_pack4,
    global_load,
    mem_realtime,
    memrealtime,
    mfma_f8f8_k128,
    mul_unpacked,
    popcount,
    read_lane_i32,
    spin_pause,
    wave_or_dpp,
    wave_umax_dpp,
    write_lane_i32,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    exp as _exp,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    fp8_roundtrip as _fp8_roundtrip,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    fp8_to_bf16x8 as _fp8_to_bf16x8,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    mfma_bf16 as _mfma_bf16,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    mxfp4_to_bf16x8 as _mxfp4_to_bf16x8,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    rcp as _rcp,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    rsq as _rsq,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    rsrc as _rsrc,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    uniform as _uniform,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    uniform_f32 as _uniform_f32,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    xred as _xred,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    xshfl as _xshfl,
)
from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.plan.tiles import (
    atom_mxfp4_scale_index,
    atom_mxfp4_scale_pair,
)

from .geometry import (
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    NEG,
    POLL_MAX,
    THREADS,
    TL_COLS,
)
from .geometry_config import (
    EPS,
    FP8_MAX,
    HIDDEN,
    INDEX_FP8_MAX,
    INDEX_ROPE,
    INTER,
    KV_LORA,
    KV_SHARE,
    ML4_P_IQ_B,
    ML4_P_IQ_W,
    ML4_P_ISS_B,
    ML4_P_ISS_S,
    ML4_P_SS_B,
    ML4_P_SS_S,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    Q_PE_DIM,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
    AttentionWeight,
    KvCacheLayout,
    as_kv_cache_layout,
)
from .layout import (
    BLOCKS,
    DCP_SUMMARY_PAIRS,
    INDEX_CP_TILE,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_LOCAL_LDS_CAPACITY,
    INDEX_Q_ROWS,
    INDEX_RADIX_BINS,
    INDEX_RADIX_DIGITS,
    INDEX_TILE,
    N_QKV_A,
    N_ROUTER,
    N_ROW_TILES,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    SHARED_EXPERT_ARGS,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
    XQ_BLOCKS,
    XQ_GROUPS,
    XQ_WAVES,
    cache_per_row,
    dcp_local_uv_tile,
    dcp_summary_index,
    dcp_uv_owner,
    dn_tile,
    down_prefetch_batch,
    down_x_words,
    fp8_kv_upper_pair_lane,
    fp8_pe_upper_pair_lane,
    index_cp_capacity,
    index_cp_partitions,
    index_cp_world,
    index_fan,
    index_fp8_data_dword,
    index_fp8_scale_dword,
    index_levels,
    index_lists,
    index_seg,
    index_select_capacity,
    index_select_stage,
    indexer_uses_tiles,
    layout,
    ptpc_stage_chunk,
    qkv_index_tasks,
    qkv_sample_tiles,
    sample_wave_batches,
    sparse_keys_per_task,
    split_acc_head,
    split_score_column,
    stage_tasks,
    ug_split,
)

# The index K writer's scale is 2 ** (exponent(amax) - INDEX_FP8_EXP), plus
# one above a 1.75 mantissa: INDEX_FP8_MAX = 1.75 * 2 ** INDEX_FP8_EXP.
INDEX_FP8_EXP = round(math.log2(INDEX_FP8_MAX / 1.75))
assert INDEX_FP8_MAX == 1.75 * 2**INDEX_FP8_EXP

# ------------------------------------------------- selection tree, per token
# ``layout``'s ``index_lists``/``index_levels`` size the tree a BUILD carries;
# these four size the part a TOKEN runs, on its traced ``bound``.  A token below
# the build's maximum therefore reaches its answer at a lower level through
# ``select``es on runtime data, never through Python control flow -- which is
# what keeps one build, and one CUDA graph, correct for every context.
# Mirrors ``atom/models/deepseek_v41/mono/index_plan.py:95-154``, ``:285-291``.
# Only non-negative operands reach ``//``, where Python's floor and the device's
# truncation agree.


def _tree_cdiv(a, b):
    return (a + (b - 1)) // b


def tree_lists(bound, level, seg, fan):
    """Lists a token's selection holds at `level`."""
    n = _tree_cdiv(fx.max(bound, fx.Int32(0)), seg)
    for _ in range_constexpr(level):
        n = _tree_cdiv(n, fan)
    return n


def tree_inputs(bound, level, j, seg, fan, k):
    """Values list `j` of `level` ranks: a segment's keys at level 0, else
    the valid entries of the lists it merges -- every one of those but the last
    holds exactly `k`, which is what makes the merged input dense."""
    if const_expr(level == 0):
        return fx.max(fx.min(bound - j * seg, fx.Int32(seg)), fx.Int32(0))
    below = tree_lists(bound, level - 1, seg, fan)
    first = j * fan
    count = fx.min(below - first, fx.Int32(fan))
    last = first + count - 1
    valid = fx.min(tree_inputs(bound, level - 1, last, seg, fan, k), fx.Int32(k))
    return fx.max(count - 1, fx.Int32(0)) * k + valid


def tree_count(bound, level, seg, fan):
    """Lists a token runs at `level`: level 0 always one (a token that sees a
    single key still emits), a higher level only while the one below had more
    than one -- so exactly one level per token is its final one."""
    if const_expr(level == 0):
        return fx.max(tree_lists(bound, 0, seg, fan), fx.Int32(1))
    return (tree_lists(bound, level - 1, seg, fan) > 1).select(
        tree_lists(bound, level, seg, fan), fx.Int32(0)
    )


def tree_final(bound, level, seg, fan):
    """Is `level`'s single list this token's answer?  It publishes from it."""
    return tree_lists(bound, level, seg, fan) <= 1


# a mailbox wait gives up after this long (s_memrealtime, 100 MHz: 10 s)
FAULT_TICKS = 10 * 100_000_000
FAULT_CHECK_SPINS = 32  # power of two


def build_ml4_monokernel(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    index_request_width: int = 1,
    index_page_size: int = 16,
    index_block_table_stride: int | None = None,
    index_cache_dtype: str = "bf16",
    expert_mxfp4: bool = False,
    atom_experts: bool = False,
    # FP8 experts read in place from aiter's block-scale MoE storage
    # (`shuffle_weight` 16x16: a 64-k chunk holds k 16 (lane // 16) .. + 16 of
    # row lane % 16 in each lane), so the FP8 activations sit in LDS in natural
    # K order instead of `f8_word` order.
    aiter_fp8_experts: bool = False,
    attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
    kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
    kv_cache_dtype: str = "bf16",
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    scale: float = SOFTMAX_SCALE,
    uv_scale_rows: int = 128,
    native_fp4_mfma: bool = False,
    rope_dtype: str = "fp32",
    timeline: bool = False,
    indexer_cp: bool = False,
    # ML4 index-sharing layers: no indexer, attend to the logical top-k the
    # previous fused-indexer launch left in the shared scratch (``indices``
    # points there); keys resolve through the block table passed as cur_pos.
    shared_indices: bool = False,
    # The shared expert's MXFP4 tensors live outside the routed-expert storage
    # (vLLM keeps 128 routed experts only): slot SHARED_EXPERT resolves to the
    # pointers at SHARED_EXPERT_ARGS.. of the parameter table (timeline arg).
    separate_shared: bool = False,
    # Fail-stop: byte offset of a sticky int32 fault word in every rank's
    # symmetric buffer. Every mailbox wait gives up after FAULT_TICKS or once
    # the word is set; giving up sets the word on every rank.
    fault_offset: int | None = None,
    # The schedule the host no longer parameterises: each of these is the arm
    # that won its measurement, and the default is now the only value ATOM
    # builds with. `moe_dedup=False` in particular is load-bearing -- grouping
    # the routed MoE measured 7.5% / 2.7% SLOWER, and this default used to be
    # True, so a caller that omitted the kwarg got the refused schedule.
    moe_dedup: bool = False,
    dn_shared_hoist: bool = True,
    split_head_merge: bool = True,
    ug_prefetch: int = 1,
    qkv_sample_tile: int = 6,
    # Each row's key count from its own position and slot (the whole context
    # up to top-k, none for a padded row), so the caller need not build
    # `sparse_kv_indptr`: the fused-indexer and index-sharing paths read their
    # top-k at a fixed row stride and use no cumulative bounds. None:
    # ``MONO_OWN_KEYS=1``.
    own_key_counts: bool | None = None,
):
    """Return the `@flyc.jit` launcher for one rank's whole layer.

    `timeline=True` records `s_memrealtime` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the `timeline` buffer:
    int64 `[sum(task counts), TL_COLS]` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in `stage_tasks` order.
    """
    assert uv_scale_rows in (64, 128)
    assert heads % WAVES == 0, "split attention maps one local head to each wave"
    assert kv_cache_dtype in ("bf16", "fp8")
    assert index_cache_dtype in ("bf16", "fp8")
    assert rope_dtype in ("fp32", "bf16")
    attention_weight = AttentionWeight(attention_weight)
    cache_layout = as_kv_cache_layout(kv_cache_layout)
    attention_bf16 = attention_weight is AttentionWeight.BF16
    attention_ptpc = attention_weight is AttentionWeight.FP8_PTPC
    use_atom_kv_cache = cache_layout is KvCacheLayout.ATOM
    if own_key_counts is None:
        own_key_counts = os.environ.get("MONO_OWN_KEYS", "0") == "1"
    if own_key_counts and not (use_atom_kv_cache and (with_indexer or shared_indices)):
        raise ValueError(
            "own_key_counts needs the ATOM cache and fixed-stride top-k "
            "(fused indexer or shared indices)"
        )
    CACHE_TASKS = S if cache_per_row() else 1
    N_QKV_IDX = qkv_index_tasks(with_indexer)
    QKV_TILES = qkv_sample_tiles(S)
    cache_fp8 = kv_cache_dtype == "fp8"
    # ATOM's production index plane; "bf16" is the private plane PR #1256 wrote.
    index_fp8 = index_cache_dtype == "fp8"
    assert not cache_fp8 or use_atom_kv_cache
    assert not atom_experts or expert_mxfp4
    assert not native_fp4_mfma or atom_experts
    assert not aiter_fp8_experts or not expert_mxfp4
    assert not GFX942 or not expert_mxfp4, (
        "MXFP4 experts need gfx950's scaled conversions and MFMA"
    )
    attention_k_chunks_per_unit = 1 if attention_bf16 else 2
    SPLIT_KEYS = sparse_keys_per_task(S, heads)
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 12
    assert 1 <= launches_per_step <= LAYER_SLOTS
    assert not with_indexer or (
        topk == 2048 and index_max_seq % INDEX_KEYS_PER_TASK == 0
    )
    assert not with_indexer or (
        S % index_request_width == 0 and index_max_seq % index_page_size == 0
    )
    # The FP8 plane preshuffles keys into MFMA 16x16 tiles, so a page is whole tiles.
    assert not (with_indexer and index_fp8) or index_page_size % INDEX_TILE == 0
    # The FP8 plane is paged and block-table addressed; SPLIT has no page to index.
    assert not index_fp8 or use_atom_kv_cache, (
        "the FP8 index plane requires the ATOM KV cache layout"
    )
    if index_block_table_stride is None:
        index_block_table_stride = index_max_seq // index_page_size
    assert (
        not with_indexer or index_block_table_stride >= index_max_seq // index_page_size
    )
    assert not (attention_bf16 and with_indexer), (
        "BF16 attention uses the external indexer"
    )
    assert not indexer_cp or with_indexer
    assert not shared_indices or (use_atom_kv_cache and not with_indexer)
    H = heads
    L = H if output_heads is None else output_heads
    W = npes
    D = dcp_size
    expert_inter = inter
    # up/gate: TOP_K * inter/8 tasks (one 8-row column tile of one routed slot each; the first inter/8 also
    # run the shared expert), in ceil(.. / 256) rounds -- whole rounds for top-8 and ML4's top-4 (1536).
    paired_ug_rows = expert_mxfp4 and not atom_experts
    assert D == 1 or (D == W and H == L * D)
    I_PER_SLOT = expert_inter // UG_TILE
    G = BLOCKS
    SC, SY = layout(
        S,
        H,
        W,
        topk,
        with_indexer,
        index_max_seq,
        indexer_cp=indexer_cp,
        inter=expert_inter,
        output_heads=L,
        dcp_size=D,
        native_fp4_mfma=native_fp4_mfma,
    )
    CP_RANKS = index_cp_world(W, S, index_max_seq) if indexer_cp else 1
    INDEX_CAPACITY = (
        index_cp_capacity(index_max_seq, CP_RANKS) if indexer_cp else index_max_seq
    )
    TILED_INDEXER = with_indexer and indexer_uses_tiles(index_max_seq, indexer_cp)
    LOCAL_INDEX_CAPACITY = min(index_max_seq, INDEX_LOCAL_LDS_CAPACITY)
    # The local selector is a merge tree over fixed ``INDEX_SEG`` leaves: the
    # leaf is what it stages in LDS, the depth is what carries the context.  A
    # tiled build keeps it only as the short-query fallback over a one-leaf-wide
    # peer score plane, so there it stays depth 0 -- exactly today's code.
    SELECT_CAPACITY = index_select_capacity(index_max_seq, indexer_cp)
    assert (
        index_seg(index_max_seq) == LOCAL_INDEX_CAPACITY
    )  # the leaf IS the LDS staging
    INDEX_FAN = index_fan(index_max_seq, topk)
    INDEX_DEPTH = index_levels(SELECT_CAPACITY, topk)
    SHARD_INDEX_Q = indexer_cp and not TILED_INDEXER and W > 1
    CP_PARTS = index_cp_partitions(index_max_seq, CP_RANKS) if TILED_INDEXER else 0
    N_SPLIT = topk // SPLIT_KEYS
    N_HEAD_GROUPS = H // WAVES
    # ---- the merged head groups (``split_head_merge``)
    # ``gather_old_kv`` copies the (sample, key-chunk) KV rows into LDS and
    # closes over nothing else: no ``head_group``, no ``h``. The two tasks
    # that share an (s, t) and differ only in head group therefore issue
    # byte-identical gathers -- 384 of the 768 at S=12, each 64 keys x
    # (KV_LORA + PE_DIM) bytes. The q wait above it is duplicated for the
    # same reason, and it already waits on all H heads, not this group's.
    # So let one task own both groups and run them BACK TO BACK.
    #
    # Sequential, not simultaneous, and that is the whole feasibility
    # argument: the PV accumulators are local to a group's body, ``red`` is
    # rewritten under a barrier, and ``pl`` is already sized for all H
    # heads. Nothing has to be live across the two, so the register
    # pressure does not double -- which it would if the groups were
    # interleaved, and this kernel has no room (239 of 256 VGPRs at S=12).
    #
    # Bit-identical outright: no value, order or rounding moves, only which
    # task computes it. The q staging goes back to covering all H heads at
    # once, which is what ``QHEADS`` narrows per group -- once over H here
    # costs exactly what twice over H/2 cost before.
    SPLIT_MERGE = bool(split_head_merge) and N_HEAD_GROUPS > 1
    SPLIT_GROUPS = N_HEAD_GROUPS if SPLIT_MERGE else 1
    N_SPLIT_TASKS = S * (N_HEAD_GROUPS // SPLIT_GROUPS) * N_SPLIT
    QB_ROWS = H * (NOPE_DIM + Q_PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    # The index-Q row space is fixed at 256 tiles, one per CTA: the q_b CTAs
    # produce the first N_QB (TP4: all of them, TP8: a quarter), the others
    # the rest.
    assert not with_indexer or (INDEX_Q_ROWS // INDEX_TILE == BLOCKS and N_QB <= BLOCKS)
    QB_PER_HEAD = (NOPE_DIM + Q_PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    N_UV = H * V_DIM // UV_TILE
    O_K = L * V_DIM
    N_UG = S * MOE_SLOTS * I_PER_SLOT
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs); row
    # strides are padded by 4 words so the MFMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_KEYS * KS
    # The input projection is processed ``qkv_sample_tile`` samples at a time.
    # Besides keeping the MFMA N dimension dense, this caps its normalized
    # activation tile (48 KiB at 4 samples for S=8).  Later stages either
    # consume a smaller tensor or use the FP8 representation and therefore fit
    # all samples at once.
    #
    # The tile is also how often the 16.1 MB input-projection weight is READ:
    # once per tile, and that weight does not depend on S at all. At 4 it is
    # read ceil(S/4) times -- twice at S=6, THREE times at S=12 -- which is
    # most of why this stage grows 1.74x between those widths (9.1 -> 15.8 us
    # measured) when a weight-bound stage should not grow at all. The MFMA has
    # 16 N columns and uses 4 of them. Raising the tile trades LDS for weight
    # traffic; it is bit-identical, because the samples are independent output
    # columns and each one's K chunks accumulate in the same order either way.
    # gfx942: four 6144-wide bf16 rows (48 KB) is all its 64 KB of LDS holds
    # next to the reductions, so wider steps also stage the q_b and index_q
    # input four samples at a time (``X_TILE``), over the same weight registers.
    SAMPLE_TILE = min(S, 4 if GFX942 else qkv_sample_tile)
    X_TILE = SAMPLE_TILE if GFX942 else S
    # A q_b / index_q step wider than X_TILE stages its input tile by tile;
    # each tile's poll loads (one batch a thread) go out under the previous
    # tile's MFMA chain and reduction instead of after them.
    _X_TILE_PRE = GFX942 and X_TILE < S and X_TILE * HIDDEN // 4 // THREADS <= POLL_MAX
    assert not (attention_ptpc and X_TILE < S)
    # Per-sample strides of the MoE FP8 activation tile and its block scales,
    # in LDS.  ``HIDDEN // 4`` (1536) and ``XQ_GROUPS`` (192) are both multiples
    # of the 32 LDS banks, so when the up/gate MFMA's B column is the SAMPLE --
    # which is what one expert serving many rows means -- all 16 columns start
    # on the same bank and the read is a 16-way conflict.  Skewing each stride
    # by one bank group spreads them; +4 words is optimal for the 16-byte
    # activation read (16 lanes x 4 banks needs two cycles regardless) and +1
    # makes the one-word scale read conflict-free.  Neither changes LDS size:
    # ``X_WORDS`` is set by sparse attention, not by this tile.
    XQ_X_STRIDE = HIDDEN // 4 + 4
    XQ_S_STRIDE = (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS) + 1
    INDEX_BF16_OFF = SAMPLE_TILE * HIDDEN // 4
    INDEX_Q_BF16_OFF = S * Q_LORA // 4
    DN_TILE = dn_tile(S, expert_mxfp4)
    N_DN_TILES = HIDDEN // DN_TILE
    # Both of these MUST live here, in ``build_ml4_monokernel``'s scope, and
    # not inside ``ml4_monokernel``: FlyDSL's compile-cache key folds in the
    # kernel's *closure* scalars (``_collect_closure_scalar_vals``), so a
    # compile-time constant that is a plain local of the kernel body does not
    # change the key and a second build silently replays the first binary.
    DN_BATCH = down_prefetch_batch(S, native_fp4_mfma, atom_experts)
    # ``MONO_DN_HOIST=0`` restores the per-unit expert resolution the
    # ``DN_HOIST`` group below replaces, for an in-process A/B. Both are
    # bit-identical; see the comment at ``dn_expert``.
    _DN_HOIST = os.environ.get("MONO_DN_HOIST", "1") == "1"
    # How many attention heads a `split` task stages into the Q tile. It reads
    # only its own head group's WAVES rows, so staging all H of them polls and
    # stores H // WAVES times the q words it uses. Narrowing it is
    # bit-identical. Requires a whole number of THREADS-wide staging passes and
    # at least one; otherwise fall back to the old all-H staging.
    # ``MONO_SPLIT_QHEADS=0`` restores it for an in-process A/B.
    # ``MONO_R2_OFF=1`` turns off every round-2 schedule change at once, so the
    # pair can be A/B'd against the previous kernel in a single interleaved
    # process. All of them are bit-identical.
    _R2_OFF = os.environ.get("MONO_R2_OFF", "0") == "1"
    # How many (sample, group) cells of ``stage_x_pairs_ptpc`` share one poll
    # batch and one ``block_maxs``. 1 restores the per-cell walk; <= 0 batches
    # every cell. Changing it is bit-identical; see ``ptpc_stage_chunk``.
    PTPC_CHUNK = 1 if _R2_OFF else ptpc_stage_chunk(S)
    _PTPC_LEGACY_POLL = _R2_OFF
    _SPLIT_QHEADS = not _R2_OFF and os.environ.get("MONO_SPLIT_QHEADS", "1") == "1"
    # ``MONO_UG_SCALE_PAIR=0`` restores the per-unit 1-byte MXFP4 scale gather
    # in the up/gate stage, for an in-process A/B. Both are bit-identical; see
    # ``atom_mxfp4_scale_pair`` and the comment in ``ug8_units``.
    _UG_SCALE_PAIR = not _R2_OFF and os.environ.get("MONO_UG_SCALE_PAIR", "1") == "1"
    # When the up/gate tasks do not fill whole CTA sweeps (TP8: 384 over 256),
    # the first ``UG_TASKS % G`` CTAs of the rotation run one task more. Task u
    # runs logical task ``(u + UG_SHIFT) % UG_TASKS``, which puts the ``inter/8``
    # tasks that also carry the shared expert on CTAs with one task fewer.
    # Only the task-to-CTA assignment moves.
    _UG_TASKS = TOP_K * (expert_inter // 8)
    _UG_EXTRA = _UG_TASKS % G
    UG_SHIFT = (
        _UG_TASKS - _UG_EXTRA
        if not moe_dedup and _UG_EXTRA and _UG_EXTRA + expert_inter // 8 <= G
        else 0
    )
    # S=1 with more up/gate tasks than CTAs: a CTA stages and routes its
    # activation once for all of its tasks and keeps every task's routed weight
    # tile in flight together, instead of one full round per task.
    # Bit-identical: each task's loads, MFMAs and sums are unchanged.
    _UG1_FUSE = GFX942 and not expert_mxfp4
    # A row whose context is at most ``topk`` keys selects every key, so its
    # sparse attention reads the keys by position: the index query, scores and
    # selection are skipped and ``split`` no longer waits on them (the shared-
    # index layers apply the same rule). The index-K cache write is unchanged.
    _DENSE_SHORT = GFX942
    # A unit's coefficient is applied with a multiply the SLP vectorizer cannot
    # pack (`mul_unpacked`). Packed, its broadcast operand made the waitcnt pass
    # wait on a pending scale load of a later unit, draining the weight prefetch
    # (S = 8 `down`: four full drains in the routed stream's tail). Same products
    # for finite values.
    _SCALAR_COEF = GFX942
    # `down` quantizes a wave's mid blocks with their max reductions
    # interleaved (`quant_scaled_many`), not one block's after another.
    _DN_QUANT_BATCH = GFX942
    # `down` quantizes each 128-block of the mid once -- block k on the CTA of
    # down tile k (mod N_DN_TILES) -- and publishes it as FP8 words and a
    # scale that every CTA copies into LDS. Each CTA used to quantize all
    # S * MOE_SLOTS * inter / 128 blocks itself: 30 a wave at S = 8, ~4 us of
    # VALU (4 cycles a wave64 op) between the last mid and the first MFMA.
    _DN_MID8 = _DN_QUANT_BATCH and not expert_mxfp4 and S > 1
    # FP8 block amax as one DPP max chain read back from lane 63. The operands
    # are |x|, whose bits order like unsigned ints, so `wave_umax_dpp` picks
    # the same maximum as the float butterfly -- whose 32 / 16 steps are LDS
    # permutes on gfx942, and LLVM serialises them: the S = 8 `down` quant of
    # 30 blocks a wave took 4-6 us.
    _QUANT_DPP = GFX942
    # Float max / min without NaN propagation: gfx942 has no IEEE maximum, so
    # each one is a max, an ordered compare and a select. The operands are FP8
    # quant |x| and x * inv against +-FP8_MAX, softmax and router maxima over
    # finite logits (or -1e30 fill), |x| maxima, clamps, and the index ReLU on
    # MFMA sums (never -0: the accumulator starts at +0), where signed zeros
    # cannot change the result; finite inputs give the same bits.
    _MAX_FM = "nnan,nsz" if GFX942 else None
    # timeline stamps only in these stages (comma list); the rest build as without.
    # A tuple, not a list: the JIT cache key takes scalar and tuple closure values.
    _TL_ONLY = tuple(s for s in os.environ.get("MONO_TL_ONLY", "").split(",") if s)
    # and only these columns (comma list of ints), e.g. 0,4 for start and end
    _TL_COLS = tuple(int(c) for c in os.environ.get("MONO_TL_COLS", "").split(",") if c)
    # A split past its row's last key (a dense row shorter than ``topk``)
    # holds no keys: it is skipped, and the merge reads only the splits that
    # hold keys. A skipped split's weight exp(NEG - M) / L is exactly 0, so
    # its partial sum never contributed. Split 0 always runs (it publishes a
    # padded row's empty softmax).
    #
    # Split tasks run in row-major (sample, head group, split) order, so a
    # CTA's task one round later is the same split of a later row. Odd rounds
    # then walk the splits in reverse (`split_coords`): a short row's live
    # splits are the low ones, and a CTA pairs a low split with a high one
    # instead of two live ones while other CTAs get two empty ones. Each
    # (sample, split) computes the same: bit-identical.
    _SKIP_EMPTY_SPLITS = GFX942 and D == 1
    # A split task's row bounds, and -- in a dense row, its keys positions
    # t * SPLIT_KEYS.. on one page -- the page of its keys (and page 0, a dead
    # key's), depend on launch inputs alone: each CTA loads them for its split
    # rounds at kernel start, with the rows' cache slots `patch_new_kv`
    # matches, as one word per lane. The split then waits on the KV gather
    # only, not on two round trips before it.
    _SPLIT_PRE = (
        _SKIP_EMPTY_SPLITS
        and use_atom_kv_cache
        and (with_indexer or shared_indices)
        and index_page_size % SPLIT_KEYS == 0
        and (N_SPLIT_TASKS + G - 1) // G <= 8
    )
    # A split's KV gather issues all of a wave's key row loads before it waits
    # on any (`gather_old_kv_batched`); at S <= 2 that measured slower.
    _GATHER_BATCH = GFX942 and S > 2
    # A split wave holding rows of this launch polls all their kvnew / penew
    # pairs at once (`patch_new_kv_batched`); at S = 1 (one new key at most)
    # that measured slower.
    _PATCH_BATCH = GFX942 and S > 1
    # A split's QK reads each K step's LDS operands during the previous
    # step's MFMAs, not after them.
    _QK_PIPE = GFX942
    # The four W_UV tasks of a head (UV_TILE = 32) each merge one latent
    # quarter of the splits' partial sums, publish it (`lat`) and gather the
    # other three, instead of each merging the whole latent: a merge pulls a
    # quarter of the bytes through the (L2-bypassing) coherent loads, which
    # bound it.
    _UV_SLICE = (
        GFX942 and D == 1 and not attention_ptpc and UV_TILE * KV_LORA == V_DIM * 128
    )
    # The shared expert's four weight pointers are read from the parameter
    # table once, ahead of the MoE, instead of next to each use. Measured
    # (same-GPU ABBA): -1.1 us at S = 2, but +0.5 us at S = 1 and +1.6 us at
    # S = 4, so it is on at S = 2 only.
    _HOIST_SHARED = GFX942 and S == 2
    # `down` reuses the routing table `ug` left in LDS instead of polling the
    # scores again and re-running the top-k: at S > 1 `ug` starts with
    # `dn_route` itself, and the fused S = 1 path writes the same table from
    # its own picks. The re-route cost ~3.5 us on the CTAs that finish `ug`
    # last, which is where `down`'s critical path starts.
    _DN_ROUTE_REUSE = GFX942 and (
        S > 1 or (_UG1_FUSE and TOP_K * (expert_inter // 8) > BLOCKS)
    )
    # S > 1 up/gate as one flat task list (shared-expert units first, then
    # every sample's routed units) striped over the CTAs with a one-unit
    # prefetch across task boundaries, instead of per-sample rounds that each
    # restart the weight pipeline. -0.5 us at S = 2, even at S = 4;
    # bit-identical.
    _UG_FLAT = GFX942 and not expert_mxfp4 and not moe_dedup and not atom_experts
    # The flat list's shared-expert units (one per CTA, on the first
    # expert_inter // 8 CTAs of the rotation) need no routing: those CTAs run
    # theirs while the router is still busy, before `dn_route`, so the
    # routed units -- a whole number of rounds at even S -- are all that is
    # left once the scores land. Otherwise the shared CTAs carry one unit
    # more than the rest, which streams as a partly empty last round.
    _UG_SH_EARLY = _UG_FLAT and expert_inter // 8 <= G
    # The flat list's routed units name each DISTINCT expert once, with every
    # sample in the MFMA's N columns as the shared unit has them, and a sample
    # keeps the columns of the experts it routes to. A column's value does not
    # depend on the others, so this is bit-identical. The L2 / MALL already
    # serve most repeated tiles; what it saves is the rest of a repeat's cost.
    # Measured with GSM8K-like overlap (22 distinct experts in 32 picks at
    # S = 8, 13 in 16 at S = 4): -7.7 us at S = 8 on one GPU, -6.7 at TP8;
    # nothing at S = 4, and +1.5 us at S = 8 with no overlap (one GPU). On at
    # S >= 8.
    _UG_DEDUP = _UG_SH_EARLY and S >= 8
    # Streams that carry only routed picks (the flat up/gate list past the
    # shared units, `down` once `DN_SHARED` hoists slot 0 out) address their
    # experts without the separate shared expert's select (`expert_addr`), so
    # a tile's loads go out while the previous tile's are still in flight.
    # Measured at TP8: -2.3 us at S = 8, but +0.4 at S = 2 (-0.7 on one GPU);
    # +1.1 / +0.8 at S = 4 on one GPU. On at S >= 8.
    _ROUTED_ADDR = GFX942 and S >= 8
    QHEADS = (
        WAVES
        if (
            _SPLIT_QHEADS
            and heads % WAVES == 0
            and (WAVES * KV_LORA // 4) % THREADS == 0
        )
        else heads
    )
    RED_WORDS = WAVES * 64 * 4
    LDS_KEYS = max(264 if with_indexer else 0, SPLIT_KEYS, S * MOE_SLOTS)

    # TileRT lineage: use one phase-overlaid arena instead of summing every
    # stage's LDS requirement.  The Kimi kernel reuses this same fusion pattern.
    # The largest X users are the four-sample input projection, all-sample MoE FP8
    # activations, and sparse attention.  Metadata, reductions, and outputs live
    # after that common X region because they are simultaneously live in GEMVs.
    SPLIT_X_WORDS = PT_OFF + SPLIT_KEYS * PS
    index_lds_words = (
        LOCAL_INDEX_CAPACITY
        if with_indexer and (not TILED_INDEXER or indexer_cp)
        else 0
    )
    X_WORDS = max(
        SAMPLE_TILE * HIDDEN // 2,
        X_TILE
        * HIDDEN
        // 2,  # ML4: q_b and index_q stage the full normed hidden (no q-LoRA)
        S * XQ_X_STRIDE,
        SPLIT_X_WORDS,
        index_lds_words,
        (INDEX_BF16_OFF + SAMPLE_TILE * HIDDEN // 2)
        if with_indexer and attention_ptpc
        else 0,
        (INDEX_Q_BF16_OFF + S * Q_LORA // 2) if with_indexer and attention_ptpc else 0,
        down_x_words(S, expert_inter, expert_mxfp4, native_fp4_mfma),
    )
    MISC_OFF = X_WORDS
    MISC_WORDS = max(
        8 + S * XQ_S_STRIDE,
        S * MOE_SLOTS * (expert_inter // (32 if native_fp4_mfma else 128)),
        N_SPLIT + 2 + (4 if attention_ptpc else 0),
    )
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE)
    # ---- expert-group table (``moe_dedup``)
    # The routed (sample, slot) units of one step name only ``UG_GROUPS``
    # DISTINCT experts, so the up/gate stage visits each expert once and gives
    # its rows the MFMA's own N columns instead of re-reading the weight tile
    # per row.  ROUTED units at a time: a lead flag, a member mask (one nibble
    # per sample, slot + 1, zero = this sample does not route here), then the
    # compacted (expert, mask) list the task loop walks, then the count.
    UG_ROUTED = S * TOP_K
    UG_MASK_WORDS = (S * 4 + 31) // 32
    UG_GROUP_WORDS = (
        UG_ROUTED * (2 + 2 * UG_MASK_WORDS) + 1 if moe_dedup and S > 1 else 0
    )
    if _UG_DEDUP:  # `ug_flat`'s group -> expert map
        UG_GROUP_WORDS = UG_ROUTED
    GRP_OFF = OUT_OFF + OUT_WORDS
    WORK_WORDS = GRP_OFF + UG_GROUP_WORDS
    # 40960 words = 163,840 B, which is what the device reports as usable:
    # `torch.cuda.get_device_properties(0).shared_memory_per_block == 163840`
    # on gfx950. This read 32768 (131,072 B) -- a 128 KB cap 32 KB BELOW the
    # hardware, and the fused indexer missed it at S=12 by **32 bytes**
    # (131,104 B, WORK_WORDS 32776). That silently cost batch-2 MTP5 the whole
    # fused-indexer path, and with it context parallelism, which cuts the index
    # task table 71-75% and the 1M-capacity spill from 35 VGPRs to 4.
    #
    # Raising it changes no occupancy: 114 KB and 131 KB are both one workgroup
    # per CU (two would need <= 81,920 B). And `fused_indexer_refusal` has no
    # width check, so hitting this assert was never a graceful fallback -- it
    # is an AssertionError, which the runner's `except (MonoUnsupported,
    # ValueError)` does not catch.
    # A ValueError, not an `assert`: this is "that configuration does not fit",
    # which the mono runner already handles by falling back to the external
    # indexer (`except (MonoUnsupported, ValueError)`). An AssertionError is
    # not in that tuple, so the overrun used to take the engine down instead --
    # and `fused_indexer_refusal` has no width check to catch it earlier, so
    # the fused path at an unsupported width was a crash, not a fallback.
    # gfx942 has 64 KB of LDS a CU.
    LDS_WORDS = 16384 if GFX942 else 40960
    if WORK_WORDS > LDS_WORDS:
        raise ValueError(
            f"static LDS {WORK_WORDS * 4} B exceeds the per-workgroup budget "
            f"of {LDS_WORDS * 4} B at S={S}"
        )

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(
        S,
        H,
        topk,
        with_indexer,
        index_max_seq,
        expert_mxfp4,
        indexer_cp=indexer_cp,
        npes=W,
        inter=expert_inter,
        split_head_merge=bool(split_head_merge) and H // WAVES > 1,
    ):
        first[name] = acc
        acc += n
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(
        stage_tasks(
            S,
            H,
            topk,
            with_indexer,
            index_max_seq,
            expert_mxfp4,
            indexer_cp=indexer_cp,
            npes=W,
            inter=expert_inter,
            split_head_merge=bool(split_head_merge) and H // WAVES > 1,
        )
    )
    acc = 0
    for name in (
        "qkv_a",
        "q_norm",
        "cache",
        "q_b",
        "split",
        "uk",
        "uv",
        "o",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % G
        acc += tasks[name]
    if with_indexer:
        # TP8 has index-Q tiles beyond q_b's row space.  TP4 produces the full
        # index-Q row space alongside its q_b tile on every CTA.
        base["index_q"] = (base["q_b"] + N_QB) % G
        base["index_score"] = 101
        # Every tree level rotates off the same CTA the single flat select used.
        for level in range(INDEX_DEPTH + 1):
            base[index_select_stage(level, INDEX_DEPTH)] = 100
        if TILED_INDEXER:
            for digit in range(INDEX_RADIX_DIGITS):
                base[f"index_hist_{digit}"] = 101
                base[f"index_prefix_{digit}"] = 100
            base["index_count"] = 101
            base["index_offsets"] = 100
            base["index_emit"] = 101

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def ml4_monokernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        # expert-group table: lead flags, per-unit masks, then the compacted
        # (expert, mask) list and its length.  Int32-typed like ``keys``.
        grp = fx.recast_iter(fx.Int32, xs + GRP_OFF)
        GRP_MASK = UG_ROUTED
        GRP_EXP = GRP_MASK + UG_ROUTED * UG_MASK_WORDS
        GRP_GMASK = GRP_EXP + UG_ROUTED
        GRP_N = GRP_GMASK + UG_ROUTED * UG_MASK_WORDS
        pl = xs  # score Q is dead before split probabilities are written
        attn_keys = keys
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = _rsrc(h_in)
        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step (graph friendly); ``layer``
        # makes it unique per layer within the step.
        step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1
        peer_slot = (step_value * launches_per_step + layer) & 1
        # In fused mode cur_pos carries the per-request page table.  ATOM cache
        # rows and selected attention keys both use physical page slots.
        if const_expr(with_indexer and use_atom_kv_cache):
            pos0 = fx.Int32(0)
        else:
            pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32))
        r_peers = _rsrc(peers)
        if const_expr(fault_offset is not None):
            r_fault = _rsrc(sym + fx.Int64(fault_offset))

        def raise_fault():
            """Set the fault word on every rank (their waits then end at once)."""
            for p in range_constexpr(npes):
                pw = fx.Vector(bo.buffer_load(r_peers, p * 2, vec_width=2, dtype=T.i32))
                base = (fx.Int64(_uniform(pw[1])) << 32) | fx.Int64(
                    fx.Uint32(_uniform(pw[0]))
                )
                bo.buffer_store(
                    fx.Int32(1),
                    _rsrc(base + fx.Int64(fault_offset)),
                    0,
                    cache_modifier=CM_SYS,
                )

        # One wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(
            bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32)
        )
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(_uniform(pv[0]))
        )

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(
                fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16))
            )

        def fmax(a, b):
            return fx.max(a, b, fastmath=_MAX_FM)

        def fmin(a, b):
            return fx.min(a, b, fastmath=_MAX_FM)

        def _fp8_sat(x):
            return fmin(fmax(x, -FP8_MAX), FP8_MAX)

        def ld_fp8_pair(r, i):
            """E4M3 bytes `2 * i` and `2 * i + 1` of `r` (exact, scale 1) as f32."""
            word = bo.buffer_load(r, i // 2, vec_width=1, dtype=T.i32)
            pair_type = fx.Vector.make_type(2, fx.Float32)
            lo = fx.Vector(
                rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False)
            )
            hi = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=True))
            upper = (i % 2) == 1
            return upper.select(hi[0], lo[0]), upper.select(hi[1], lo[1])

        def ld_fp8(r, i):
            """E4M3 byte `i` of `r` (exact, scale 1) as f32."""
            v0, v1 = ld_fp8_pair(r, i // 2)
            return ((i % 2) == 1).select(v1, v0)

        def ld_rope(r, i):
            if const_expr(rope_dtype == "bf16"):
                return ld_bf16(r, i)
            return ld_f32(r, i)

        # Row k's word of a per-row launch input (positions and the slot
        # mapping are int64: every other word), or with ``pre`` its copy in
        # lane PRE_TABLES.index(name) * 16 + k of ``row_pre`` (`_SPLIT_PRE`;
        # k uniform).
        PRE_TABLES = ("positions", "slot_mapping", "indptr", "pages")

        def ld_row(name, k, pre=False):
            if const_expr(pre):
                return read_lane_i32(row_pre, PRE_TABLES.index(name) * 16 + k)
            t, stride = {
                "positions": (positions, 2),
                "slot_mapping": (slot_mapping, 2),
                "indptr": (sparse_kv_indptr, 1),
            }[name]
            return fx.Int32(
                bo.buffer_load(_rsrc(t), k * stride, vec_width=1, dtype=T.i32)
            )

        def row_index_bounds(s, pre=False):
            if const_expr(own_key_counts):
                position = ld_row("positions", s, pre)
                count = (row_slot(s, pre) >= 0).select(
                    fx.min(position + 1, fx.Int32(topk)), fx.Int32(0)
                )
                return fx.Int32(0), count
            return ld_row("indptr", s, pre), ld_row("indptr", s + 1, pre)

        def row_active(s, pre=False):
            if const_expr(use_atom_kv_cache):
                begin, end = row_index_bounds(s, pre)
                return end > begin
            return True

        def row_position(s, pre=False):
            if const_expr(use_atom_kv_cache):
                position = ld_row("positions", s, pre)
                present = row_active(s, pre) | (row_slot(s, pre) >= 0)
                return present.select(position, fx.Int32(0))
            return pos0 + s

        def long_row(s, pre=False):
            """Row s has more than ``topk`` keys, so its top-k is a real selection."""
            if const_expr(not _DENSE_SHORT):
                return fx.Boolean(True)
            return row_position(s, pre) + 1 > topk

        def any_long_row():
            any_long = long_row(0)
            for s in range_constexpr(1, S):
                any_long = any_long | long_row(s)
            return any_long

        def row_slot(s, pre=False):
            if const_expr(use_atom_kv_cache):
                return ld_row("slot_mapping", s, pre)
            return pos0 + s

        def split_rows(s, pre=False):
            """(nkeys, sparse, use_indices, index_base) of sample s: loads only."""
            if const_expr(use_atom_kv_cache):
                index_base, index_end = row_index_bounds(s, pre)
                nkeys = index_end - index_base
                sparse = index_end > index_base
            else:
                index_base = s * topk
                kv_len = pos0 + s + 1
                sparse = kv_len > topk
                nkeys = sparse.select(fx.Int32(topk), kv_len)
            # read the selected indices, not the keys in position order
            use_indices = (
                sparse & long_row(s, pre) if const_expr(_DENSE_SHORT) else sparse
            )
            return nkeys, sparse, use_indices, index_base

        def split_coords(tt):
            """(sample, first head group, split) of split task tt."""
            if const_expr(SPLIT_MERGE):
                s = tt // N_SPLIT
                hg0 = fx.Int32(0)
            else:
                s = tt // (N_HEAD_GROUPS * N_SPLIT)
                hg0 = (tt // N_SPLIT) % N_HEAD_GROUPS
            t = tt % N_SPLIT
            if const_expr(_SKIP_EMPTY_SPLITS):  # odd rounds in reverse
                t = ((tt // G) % 2 == 1).select(fx.Int32(N_SPLIT - 1) - t, t)
            return s, hg0, t

        def row_writes_cache(s):
            return row_slot(s) >= 0

        def index_request_start(s):
            return (s // index_request_width) * index_request_width

        def index_page_cell(s, logical_pos):
            """(physical page, row within the page) of one logical index position."""
            request = s // index_request_width
            page = fx.Int32(
                bo.buffer_load(
                    _rsrc(cur_pos),
                    request * index_block_table_stride + logical_pos // index_page_size,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            return page, logical_pos % index_page_size

        def index_page_slot(s, logical_pos):
            if const_expr(not use_atom_kv_cache):
                return logical_pos
            page, row = index_page_cell(s, logical_pos)
            return page * index_page_size + row

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return (
                fx.Vector.from_elements([a, b], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Float32)[0]
            )

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def ml4_ssmax(position, s_off, b_off, head):
            """ML4 SSMax query scale for one head, with the reference's bf16 roundings: bf16(bf16(log(pos+1)) * s) + b."""
            r_p = _rsrc(g_q)
            lp = bf16_round(fmath.log(fx.Float32(position) + 1.0))
            sc = bf16_round(
                bf16_round(lp * bf16_round(ld_f32(r_p, s_off + head)))
                + bf16_round(ld_f32(r_p, b_off + head))
            )
            return sc

        def index_arg(i):
            """Load one uniform pointer from the compact indexer parameter table.

            Fused-indexer launches carry this table in the otherwise independent
            timeline argument, keeping the no-indexer kernel ABI identical to the
            original layer.  The indices argument similarly carries index_cache.
            """
            pv = fx.Vector(
                bo.buffer_load(_rsrc(timeline_buf), i * 2, vec_width=2, dtype=T.i32)
            )
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
                fx.Uint32(_uniform(pv[0]))
            )

        shared_ptrs = {}

        def shared_ptr(which):
            """The separate shared expert's tensor `which` (see `expert_addr`)."""
            got = shared_ptrs.get(which)
            if const_expr(got is None):
                return index_arg(SHARED_EXPERT_ARGS + which)
            return got

        def hoist_shared_ptrs():
            """Read the shared expert's pointers once, ahead of the MoE weight
            streams. `vmcnt` retires in order, so a param-table load issued behind
            in-flight weight tiles waits for all of them before its address is
            usable, and the next tile's loads cannot be issued ahead of time."""
            if const_expr(separate_shared and _HOIST_SHARED):
                for which in range_constexpr(4):
                    shared_ptrs[which] = index_arg(SHARED_EXPERT_ARGS + which)

        def expert_addr(base, e, stride, which, routed_only=False):
            """Address of expert `e` in one expert tensor (`which`: 0 w_ug, 1 s_ug, 2 w_dn, 3 s_dn).

            `routed_only`: `e` is a routed pick, never the separate shared
            expert, whose pointer is a param-table load. `vmcnt` retires in
            order, so that load (selected away or not) waits for every weight
            tile in flight, and the next tile's loads issue only after it."""
            routed = base + fx.Int64(e) * fx.Int64(stride)
            if const_expr(separate_shared and not routed_only):
                return (fx.Int32(e) == fx.Int32(SHARED_EXPERT)).select(
                    shared_ptr(which), routed
                )
            return routed

        # ---- tagged-pair mailboxes
        # TileRT lineage: payload + launch epoch is the progress protocol for
        # resident CTAs; the helpers below are the FlyDSL/ROCm adaptation.
        def mb(name):
            return scratch + fx.Int64(SC[name])

        def cp_local_rank():
            if const_expr(indexer_cp and CP_RANKS < W):
                return rank % CP_RANKS
            return rank

        def cp_peer(peer):
            if const_expr(not indexer_cp):
                return sym
            if const_expr(CP_RANKS < W):
                peer = (rank // CP_RANKS) * CP_RANKS + peer
            words = fx.Vector(
                bo.buffer_load(r_peers, peer * 2, vec_width=2, dtype=T.i32)
            )
            return (fx.Int64(_uniform(words[1])) << 32) | fx.Int64(
                fx.Uint32(_uniform(words[0]))
            )

        def cp_buffer(name, address):
            return (
                address
                + fx.Int64(SY[name])
                + fx.Int64(peer_slot) * SY[name + "_stride"]
            )

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); `v` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(
                fx.Vector.from_elements([bits, tag], fx.Int32),
                _rsrc(base_addr),
                i * 2,
                cache_modifier=cm,
            )

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [
                    fx.Float32(v0).bitcast(fx.Int32),
                    tag,
                    fx.Float32(v1).bitcast(fx.Int32),
                    tag,
                ],
                fx.Int32,
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store.
            """
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                _rsrc(base_addr),
                i,
                cache_modifier=cm,
            )

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(
                fx.Float32
            )

        def _qptr(addr):
            return fx.inttoptr(
                fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                fx.Int64(addr),
            )

        def _ld_pair(addr, scope):
            """One (value, tag) pair as a single 64-bit relaxed atomic load: never hoisted,
            coherent at `scope` (agent -> sc1, system -> sc0 sc1)."""
            return fx.generic_load(
                _qptr(addr), memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope
            )

        def poll_issue(specs, scope="agent"):
            """`poll`'s first loads of `specs` (at most POLL_MAX), issued now and
            waited for by ``poll(specs, first=...)``."""
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS
            words = []
            for b, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(
                        _rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        def poll(specs, scope="agent", batch=POLL_MAX, first=None):
            """Batched poll of mailbox pairs: `specs` = [(base_addr, pair index, npairs in {1, 2})].

            All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
            (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
            the whole batch is re-loaded, so a batch costs one round trip after its
            last producer lands.  A side-effecting (compiler-opaque) asm statement in
            the retry loop keeps the loads from being hoisted.  Returns one list of
            Int32 value bits per spec. `first`: the ``poll_issue(specs)`` loads,
            checked before any re-load."""
            if const_expr(len(specs) == 0):
                return []
            assert first is None or len(specs) <= batch
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(
                    specs[batch:], scope, batch
                )

            def load_all():
                return poll_issue(specs, scope)

            nw = sum(2 * n for _, _, n in specs)

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            v = load_all() if const_expr(first is None) else first
            if const_expr(fault_offset is None):
                while pending(v):
                    spin_pause()
                    v = load_all()
            else:
                # no clock read on entry (s_memrealtime is a waited scalar load):
                # the deadline starts at the first check, FAULT_CHECK_SPINS polls in
                t0 = fx.Int64(0)
                late = fx.Int32(0)
                spins = fx.Int32(0)
                while pending(v) & (late == 0):
                    spin_pause()
                    v = load_all()
                    spins = spins + 1
                    # the fault word and the clock every FAULT_CHECK_SPINS polls:
                    # the hot path of a short wait stays the plain re-poll
                    if (spins & (FAULT_CHECK_SPINS - 1)) == 0:
                        now = memrealtime()
                        t0 = (t0 == 0).select(now, t0)
                        fault = fx.Int32(
                            bo.buffer_load(
                                r_fault,
                                0,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_SYS,
                            )
                        )
                        late = ((fault != 0) | (now - t0 > FAULT_TICKS)).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                if late != 0:
                    raise_fault()
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def hint_wait(n, addr_of, mark=None):
            """Consumers poll their payload directly (tight per-wave spins); a wave-0
            pre-poll of each producer's last pair only added a hop of latency."""
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 1)
            gpu.barrier()

        def pre_poll(n, addr_of):
            """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
            before a large payload poll, so waiting CTAs do not flood memory."""
            if wave == 0:
                b, i = addr_of(fx.min(lane, n - 1))
                poll([(b, i, 1)])
            gpu.barrier()

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf(base_addr, i):
            return get(base_addr, i).bitcast(fx.Float32)

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [
                v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])
            ]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [
                (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32))
                for v in poll([(b, i, 2) for b, i in specs])
            ]

        def get2(base_addr, i):
            return get2_many([(base_addr, i)])[0]

        def get_bf2_many(specs):
            """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
            return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def wave_max(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, fmax)
            return v

        def block_sums(vs):
            """Block-wide sums of several per-thread values with one LDS exchange."""
            ws = [wave_sum(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = t + lds_ld(red, i * WAVES + w)
                tots.append(t)
            gpu.barrier()
            return tots

        def block_maxs(vs):
            """Block-wide maxima of several per-thread values with one LDS exchange."""
            ws = [wave_max(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = fmax(t, lds_ld(red, i * WAVES + w))
                tots.append(t)
            gpu.barrier()
            return tots

        def block_sum(v):
            w = wave_sum(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return t

        # ------------------------------------------------ MFMA GEMV machinery
        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None):
            """Issue one 64-k chunk of row group `rg` of a packed FP8 matrix; the
            bf16 activation chunk starts at LDS word `b_word`."""
            wv = fx.Vector(
                bo.buffer_load(
                    w_rsrc, ((rg * NKC + kc) * 64 + lane) * 4, vec_width=4, dtype=T.i32
                )
            )
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            if const_expr(callable(coef)):  # factor known only after a later wait
                return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8", [wv], s, b_word + (lane // 16) * 4)

        def unit_fp8x2(
            w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, scale_rows=SCALE_BM
        ):
            """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + lane) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
            if const_expr(callable(coef)):
                return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

        def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
            """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
            `rg` against the FP8 activation of LDS words `b_word` + [0, 32) (`f8_word`
            order); `coef()` = activation block scale (times route weight).  `ln`
            = the lane whose weights are loaded (default: own lane)."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
            return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

        def unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
            """Issue one packed 128-K PTPC FP8 tile; output-channel scale is applied after reduction."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            return ("ptpc", wv, coef, b_word + (lane // 16) * 4)

        def unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
            """Issue one packed 64-K PTPC FP8 tile."""
            ln = lane if ln is None else ln
            weight = fx.Vector(
                bo.buffer_load(
                    w_rsrc, ((rg * NKC + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32
                )
            )
            return ("ptpc64", [weight], coef, b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            """Issue one native packed 128-K MXFP4 tile and four E8M0 row scales."""
            ln = lane if ln is None else ln
            row = rg * 16 + ln % 16
            if const_expr(atom_experts):
                g4 = ln // 16
                raw = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            bo.buffer_load(
                                w_rsrc,
                                (rg * (K // 64) + kc * 2 + sp // 2) * 128
                                + (sp % 2) * 64
                                + (row % 16) * 4
                                + g4,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        for sp in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                scales = [
                    (
                        fx.Int32(
                            bo.buffer_load(
                                s_rsrc,
                                atom_mxfp4_scale_index(row, kc * 4 + sp, K // 32),
                                vec_width=1,
                                dtype=T.i8,
                            )
                        )
                        << fx.Int32(23)
                    ).bitcast(fx.Float32)
                    for sp in range_constexpr(4)
                ]
            else:
                raw = fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * (K // 128) + kc) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                packed_scale = fx.Int32(
                    bo.buffer_load(
                        s_rsrc,
                        (rg * (K // 128) + kc) * 16 + ln % 16,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                scales = [
                    (
                        (packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF))
                        << fx.Int32(23)
                    ).bitcast(fx.Float32)
                    for sp in range_constexpr(4)
                ]
            return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

        def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            fmt, weights, factor, _ = unit_mxfp4(
                w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln
            )
            return ("mxfp4_bf16", weights, factor, b_word + (lane // 16) * 4)

        def unit_native_mxfp4(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            K,
            b_word,
            b_scale_word,
            coef=None,
            ln=None,
            scale=None,
        ):
            """Issue ATOM's 16-byte/lane W4 tile directly on the scaled MFMA.

            `scale` lets the caller supply the MXFP4 weight scale byte it has
            already loaded (see `atom_mxfp4_scale_pair`)."""
            ln = lane if ln is None else ln
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            if const_expr(scale is None):
                scale = fx.Int32(
                    bo.buffer_load(
                        s_rsrc,
                        atom_mxfp4_scale_index(
                            rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32
                        ),
                        vec_width=1,
                        dtype=T.i8,
                    )
                )
            return ("native_mxfp4", (raw, scale, b_scale_word), coef, b_word)

        def lds_mxfp8(chunk_word):
            """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
            words = [
                fx.Vector(
                    fx.ptr_load(
                        xs + chunk_word + 16 * half + 4 * (lane // 16), result_type=v4f
                    )
                ).bitcast(fx.Int32)
                for half in range(2)
            ]
            return fx.Vector.from_elements(
                [words[half][e] for half in range(2) for e in range(4)], fx.Int32
            )

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def unit_attention(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            NKC,
            K,
            BK,
            b_word,
            ln=None,
            scale_rows=SCALE_BM,
            coef=None,
        ):
            if const_expr(attention_bf16):
                return unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln)
            if const_expr(attention_ptpc):
                if const_expr(BK == 64):
                    return unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
                return unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
            if const_expr(BK == 64):
                return unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word)
            return unit_fp8x2(
                w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, scale_rows=scale_rows
            )

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(fmt in ("mxfp4", "mxfp4_bf16")):
                    raw, scales = wv
                    for sp in range_constexpr(4):
                        a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                        if const_expr(fmt == "mxfp4"):
                            wh, ws = sp // 2, sp % 2
                            bv = fx.Vector(
                                fx.ptr_load(xs + (bw + wh * 16), result_type=v4f)
                            ).bitcast(fx.Int32)
                            b = _fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                        else:
                            b = fx.ptr_load(
                                xs + (bw + sp * 16), result_type=v4f
                            ).bitcast(fx.BFloat16)
                        c = _mfma_bf16(a, b, c)
                if const_expr(fmt == "f8f8"):  # one 128-k FP8 x FP8 product, unscaled
                    a = fx.Vector.from_elements(
                        [wv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    bv = [
                        fx.Vector(
                            fx.ptr_load(xs + (bw + h * 16), result_type=v4f)
                        ).bitcast(fx.Int32)
                        for h in range(2)
                    ]
                    b = fx.Vector.from_elements(
                        [bv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    c = mfma_f8f8_k128(a, b, c)
                if const_expr(fmt == "native_mxfp4"):
                    raw, weight_scale, activation_scale_word = wv
                    activation_scale = lds_ld(misc, activation_scale_word).bitcast(
                        fx.Int32
                    )
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                raw,
                                lds_mxfp8(bw),
                                c,
                                4,
                                0,
                                0,
                                weight_scale,
                                0,
                                activation_scale,
                            ],
                        )
                    )
                if const_expr(fmt in ("ptpc", "ptpc64")):
                    halves64 = 2 if fmt == "ptpc" else 1
                    for half64 in range_constexpr(halves64):
                        a = wv[half64].bitcast(fx.Int64)
                        b = fx.Vector(
                            fx.ptr_load(xs + (bw + half64 * 16), result_type=v4f)
                        ).bitcast(fx.Int64)
                        for half32 in range_constexpr(2):
                            c = fx.Vector(
                                rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    T.vec(4, T.f32), [a[half32], b[half32], c, 0, 0, 0]
                                )
                            )
                nsp = (
                    4
                    if fmt == "fp8x2"
                    else (
                        2
                        if fmt
                        not in (
                            "f8f8",
                            "ptpc",
                            "ptpc64",
                            "mxfp4",
                            "mxfp4_bf16",
                            "native_mxfp4",
                        )
                        else 0
                    )
                )
                for sp in range_constexpr(nsp):
                    if const_expr(fmt in ("fp8", "fp8x2")):
                        wh = sp // 2 if fmt == "fp8x2" else 0
                        ws = sp % 2 if fmt == "fp8x2" else sp
                        a = _fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                    else:
                        a = wv[sp].bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(
                        fx.BFloat16
                    )
                    c = _mfma_bf16(a, b, c)
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
                elif const_expr(_SCALAR_COEF):
                    acc = [acc[e] + mul_unpacked(c[e], coef) for e in range(4)]
                else:
                    acc = [acc[e] + c[e] * coef for e in range(4)]
            return acc

        def run_units(make_unit, cpw, batch, pre=None, acc0=None):
            """Software pipelined: issue batch b+1's loads before computing batch b.
            `pre` = the already-issued first batch (prefetched before a wait).
            `acc0` = an accumulator to continue, for a caller that has already
            consumed units of its own (`down`'s hoisted shared expert)."""
            acc = [fx.Float32(0.0) for _ in range(4)] if acc0 is None else acc0
            starts = list(range(0, cpw, batch))
            cur = (
                pre
                if pre is not None
                else [make_unit(c) for c in range(0, min(batch, cpw))]
            )
            for bi in range_constexpr(len(starts)):
                nxt = None
                if const_expr(bi + 1 < len(starts)):
                    n0 = starts[bi + 1]
                    nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
                acc = mma_units(acc, cur)
                cur = nxt
            return acc

        def reduce_rows(R, acc, emit, count=S):
            """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
            wpr = WAVES // R
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            gpu.barrier()
            n_out = R * 16 * count
            for i in range_constexpr((n_out + THREADS - 1) // THREADS):
                t = tid + i * THREADS
                if t < n_out:
                    rl = t % (R * 16)
                    n = t // (R * 16)
                    r = rl % 16
                    tot = fx.Float32(0.0)
                    for j in range_constexpr(wpr):
                        ww = (rl // 16) * wpr + j
                        tot = tot + lds_ld(
                            red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4
                        )
                    emit(rl, n, tot)

        def emit_out(stride, base=0):
            def f(rl, n, v):
                lds_st(outs, (base + n) * stride + rl, v)

            return f

        def gemv_x_tiles(name, units, stride, mark=None):
            """One row group of the q_b / index_q GEMV over the packed bf16 mailbox
            `name` (S rows of HIDDEN), staged `X_TILE` samples at a time against
            the same issued `units` (a callable: issued after the first staging).
            Each output column is one sample, so the tiling is bit-identical.
            (``_X_TILE_PRE``) a tile's first loads go out once the previous
            tile's have landed, under its MFMA chain and reduction."""
            nxt = None
            for sb in range_constexpr(0, S, X_TILE):
                cnt = min(X_TILE, S - sb)
                stage_x_pairs(
                    name, cnt * HIDDEN, lambda k, sb=sb: sb * HIDDEN + k, first=nxt
                )
                if const_expr(sb == 0 and mark is not None):
                    stamp(mark[0], mark[1], 2)
                gpu.barrier()
                if const_expr(callable(units)):
                    units = units()
                # after the weights: loads retire in order
                if const_expr(_X_TILE_PRE and sb + X_TILE < S):
                    sb2 = sb + X_TILE
                    nxt = poll_issue(
                        stage_x_specs(
                            name,
                            min(X_TILE, S - sb2) * HIDDEN,
                            lambda k, sb2=sb2: sb2 * HIDDEN + k,
                        )
                    )
                acc = mma_units([fx.Float32(0.0) for _ in range(4)], units)
                reduce_rows(1, acc, emit_out(stride, sb), cnt)

        def pick(values, index):
            value = values[0]
            for i in range_constexpr(1, len(values)):
                value = (index == i).select(values[i], value)
            return value

        def attention_b_word(element):
            return element // 4 if const_expr(attention_ptpc) else element // 2

        def emit_ptpc(stride, row_base, scale_rsrc, x_scales):
            def f(rl, n, v):
                lds_st(
                    outs,
                    n * stride + rl,
                    bf16_round(
                        v * pick(x_scales, n) * ld_f32(scale_rsrc, row_base + rl)
                    ),
                )

            return f

        def emit_ptpc_weight(stride, row_base, scale_rsrc):
            def f(rl, n, v):
                lds_st(
                    outs,
                    n * stride + rl,
                    bf16_round(v * ld_f32(scale_rsrc, row_base + rl)),
                )

            return f

        def emit_ptpc_scalar(stride, scale):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, bf16_round(v * scale))

            return f

        def stage_x_rmsnorm_ptpc(ld4s, n, gamma, loaded=None, count=S):
            """RMSNorm followed by production per-token FP8 quantization."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = (
                loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            )
            sums = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for value in vals[s * per + i]:
                        ss = ss + value * value
                sums.append(ss)
            totals = block_sums(sums)
            normed = []
            for s in range_constexpr(count):
                rstd = _rsq(totals[s] * (1.0 / n) + EPS)
                for i in range_constexpr(per):
                    normed.append(
                        [
                            vals[s * per + i][j] * rstd * gs[i][j]
                            for j in range_constexpr(4)
                        ]
                    )
            local_max = []
            for s in range_constexpr(count):
                amax = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for value in normed[s * per + i]:
                        amax = fmax(amax, fmath.absf(value))
                local_max.append(amax)
            maxima = block_maxs(local_max)
            scales = []
            for s in range_constexpr(count):
                scale = (maxima[s] == 0.0).select(
                    fx.Float32(1.0), maxima[s] * (1.0 / FP8_MAX)
                )
                scales.append(scale)
                reciprocal = 1.0 / scale
                for i in range_constexpr(per):
                    if const_expr(with_indexer):
                        for pair in range_constexpr(2):
                            lds_st(
                                xs,
                                INDEX_BF16_OFF + (s * n + ks[i]) // 2 + pair,
                                bf16_pair(
                                    normed[s * per + i][pair * 2],
                                    normed[s * per + i][pair * 2 + 1],
                                ),
                            )
                    values = [
                        div_rn(value, scale, reciprocal)
                        for value in normed[s * per + i]
                    ]
                    lds_st(
                        xs,
                        (s * n + ks[i]) // 4,
                        fp8_pack4(values[0], values[1], values[2], values[3]).bitcast(
                            fx.Float32
                        ),
                    )
            return scales

        def stage_x_pairs_ptpc(name, count, n, src_of, group=None, retain_bf16=False):
            """Quantize packed-BF16 mailbox rows with one scale per activation group.

            The (sample, group) cells are walked `PTPC_CHUNK` at a time. Both
            of the costs this helper used to pay per cell are **per-call**
            costs, not per-word ones: `poll` resolves a whole batch in one
            round trip after its last producer lands, and `block_maxs`
            reduces a whole *list* in two `gpu.barrier()`s. Walking one cell
            at a time therefore paid `count * groups` dependent round trips
            and twice that many barriers -- a clean O(S) term in `q_b`,
            `uk` and `o`, whose task counts and weight bytes do not depend
            on S at all. Measured with a throwaway probe that drops the
            cross-thread reduction entirely (a throwaway probe, kept at
            `mono-r2/kernel_with_diags.diff`): **9.74 us
            of a 204 us launch at S=12** and 1.35 of 114 at S=6.

            Chunking is **bit-identical**. Each list element of `block_maxs`
            reduces through its own `red` slots, so a cell's block maximum
            does not depend on how many cells share the call; the per-cell
            local maximum is taken over the same values in the same order; and
            `scales` keeps its (sample outer, group inner) order because the
            cell list is built in that order and chunked contiguously.
            """
            group = n if group is None else group
            groups = (n + group - 1) // group
            cells = []
            for s in range_constexpr(count):
                for g in range_constexpr(groups):
                    words = min(group, n - g * group) // 4
                    cells.append((s, g, words, (words + THREADS - 1) // THREADS))
            span = len(cells) if PTPC_CHUNK <= 0 else min(PTPC_CHUNK, len(cells))
            scales = []
            for c0 in range_constexpr(0, len(cells), span):
                chunk = cells[c0 : c0 + span]
                # One poll batch for the whole chunk (``poll`` splits it at
                # POLL_MAX itself, which still beats one trip per cell).
                specs = []
                for s, g, words, per in chunk:
                    for i in range_constexpr(per):
                        source_word = fx.min(tid + i * THREADS, words - 1)
                        specs.append(
                            (
                                mb(name),
                                src_of(s * n + g * group + source_word * 4) // 2,
                                2,
                            )
                        )
                if const_expr(_PTPC_LEGACY_POLL):
                    # ``MONO_R2_OFF=1``: one poll per word group, as before, so
                    # the off arm is the previous kernel exactly rather than
                    # the previous kernel plus this helper's within-cell batch.
                    got = [poll([spec])[0] for spec in specs]
                else:
                    got = poll(specs)
                at, cell_values, cell_valids, local_maxs = 0, [], [], []
                for s, g, words, per in chunk:
                    local_max = fx.Float32(0.0)
                    row_values, valids = [], []
                    for i in range_constexpr(per):
                        word = tid + i * THREADS
                        first = bf2_f32(got[at][0])
                        second = bf2_f32(got[at][1])
                        at += 1
                        values = [first[0], first[1], second[0], second[1]]
                        valid = word < words
                        row_values.append(values)
                        valids.append(valid)
                        for value in values:
                            local_max = fmax(
                                local_max,
                                valid.select(fmath.absf(value), fx.Float32(0.0)),
                            )
                    cell_values.append(row_values)
                    cell_valids.append(valids)
                    local_maxs.append(local_max)
                maxima = block_maxs(local_maxs)
                for ci in range_constexpr(len(chunk)):
                    s, g, words, per = chunk[ci]
                    scale = (maxima[ci] == 0.0).select(
                        fx.Float32(1.0), maxima[ci] * (1.0 / FP8_MAX)
                    )
                    scales.append(scale)
                    reciprocal = 1.0 / scale
                    for i in range_constexpr(per):
                        word = tid + i * THREADS
                        if cell_valids[ci][i]:
                            values = cell_values[ci][i]
                            if const_expr(retain_bf16):
                                pair_word = (s * n + g * group + word * 4) // 2
                                lds_st(
                                    xs,
                                    INDEX_Q_BF16_OFF + pair_word,
                                    bf16_pair(values[0], values[1]),
                                )
                                lds_st(
                                    xs,
                                    INDEX_Q_BF16_OFF + pair_word + 1,
                                    bf16_pair(values[2], values[3]),
                                )
                            quantized = [
                                div_rn(value, scale, reciprocal) for value in values
                            ]
                            lds_st(
                                xs,
                                (s * n + g * group + word * 4) // 4,
                                fp8_pack4(
                                    quantized[0],
                                    quantized[1],
                                    quantized[2],
                                    quantized[3],
                                ).bitcast(fx.Float32),
                            )
            return scales

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            `loaded`: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = (
                loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            )
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for a in vals[s * per + i]:
                        ss = ss + a * a
                sss.append(ss)
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 6)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 7)
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 2 + j,
                            bf16_pair(
                                a[2 * j] * rstds[s] * gs[i][2 * j],
                                a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1],
                            ),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
            gs = []
            for k in ks:
                g = (
                    fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32))
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_specs(name, n_total, src_of):
            """``stage_x_pairs``' poll specs past any remainder thread's."""
            return [
                (mb(name), src_of((tid + i * THREADS) * 4) // 2, 2)
                for i in range(n_total // 4 // THREADS)
            ]

        def stage_x_pairs(name, n_total, src_of, first=None):
            """LDS bf16 X[k] = packed bf16 mailbox `name` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements.
            `first`: ``poll_issue`` of its ``stage_x_specs``.
            """
            nq = n_total // 4
            full = nq // THREADS
            vals = poll(stage_x_specs(name, n_total, src_of), first=first)
            for i in range_constexpr(full):
                for j in range_constexpr(2):
                    lds_st(
                        xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32)
                    )
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                if w < nq:
                    v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))

        def quant_scaled(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
            amax = fmax(fmath.absf(a0), fmath.absf(a1))
            if const_expr(_QUANT_DPP):
                return quant_from_max(a0, a1, abs_wave_max(amax))
            for sh in range_constexpr(6):
                amax = _xred(amax, 32 >> sh, fmax)
            return quant_from_max(a0, a1, amax)

        def abs_wave_max(m):
            """Wave max of non-negative, non-NaN f32 lanes (wave-uniform)."""
            return wave_umax_dpp(m.bitcast(fx.Int32)).bitcast(fx.Float32)

        def quant_scaled_many(pairs):
            """quant_scaled of several blocks with their max reductions interleaved:
            each step's shuffles (LDS permutes for 32 / 16 on gfx942) overlap
            instead of running one block's chain after another."""
            amax = [fmax(fmath.absf(a0), fmath.absf(a1)) for a0, a1 in pairs]
            if const_expr(_QUANT_DPP):
                amax = [abs_wave_max(m) for m in amax]
            else:
                for sh in range_constexpr(6):
                    amax = [_xred(m, 32 >> sh, fmax) for m in amax]
            return [quant_from_max(a0, a1, m) for (a0, a1), m in zip(pairs, amax)]

        def quant_from_max(a0, a1, amax):
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(
                _rcp(amax) * FP8_MAX, fx.Float32(1.0)
            )  # hardware rcp, no IEEE divide
            q0 = fmin(fmax(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fmin(fmax(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_mxfp8(a0, a1):
            """Per-32 E4M3 values and E8M0 scale used by ATOM/AITER W4A8."""
            amax = fx.max(fmath.absf(a0), fmath.absf(a1))
            for offset in (1, 2, 4, 8):
                amax = _xred(amax, offset, fx.max)
            bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            exponent = (bits >> 23) & fx.Int32(0xFF)
            exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )
            exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
            inverse = _rcp((exponent << 23).bitcast(fx.Float32))
            q0 = fx.min(fx.max(a0 * inverse, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inverse, -FP8_MAX), FP8_MAX)
            return q0, q1, exponent

        def quant_block(a0, a1):
            """quant_scaled, values returned as the FP8-rounded f32s."""
            q0, q1, qs = quant_scaled(a0, a1)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def stage_xq(samples):
            """Poll the router's packed FP8 activation + block scales of `samples`
            (sample list, or one runtime sample) into LDS, slot 0 for one runtime sample.
            """
            nxw = HIDDEN // 4 // THREADS
            nscale = XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS
            got = poll(
                [
                    (mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1)
                    for sx in samples
                    for i in range(nxw)
                ]
                + [
                    (mb("xqs"), sx * nscale + fx.min(tid, nscale - 1), 1)
                    for sx in samples
                ]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    wd = (
                        tid + i * THREADS
                        if const_expr(native_fp4_mfma or aiter_fp8_experts)
                        else f8_word((tid + i * THREADS) * 4)
                    )
                    lds_st(
                        xs,
                        j * XQ_X_STRIDE + wd,
                        got[j * nxw + i][0].bitcast(fx.Float32),
                    )
                if tid < nscale:
                    lds_st(
                        misc,
                        8 + j * XQ_S_STRIDE + tid,
                        got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                    )

        def st_f8(k, q0, q1):
            """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
            k ^ 2) in `f8_word` order (natural order for `aiter_fp8_experts`).  Call
            from the whole wave."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            wd = k // 4 if const_expr(aiter_fp8_experts) else f8_word(k)
            if lane % 2 == 0:
                lds_st(xs, wd, (w | (nb << 16)).bitcast(fx.Float32))

        def st_f8_many(items):
            """st_f8 of several (k, q0, q1) under one lane-parity branch."""
            ws = [
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
                for _, q0, q1 in items
            ]
            nbs = [_xshfl(w, 1) for w in ws]
            wds = [
                k // 4 if const_expr(aiter_fp8_experts) else f8_word(k)
                for k, _, _ in items
            ]
            if lane % 2 == 0:
                for i in range_constexpr(len(items)):
                    lds_st(xs, wds[i], (ws[i] | (nbs[i] << 16)).bitcast(fx.Float32))

        def st_mxfp8(k, q0, q1):
            """Store one row-major MXFP8 pair for the scaled MFMA."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

        def st_ptpc(k, q0, q1):
            """Store one wave's PTPC FP8 pairs in the AITER GEMM's linear K order."""
            word = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            neighbor = _xshfl(word, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (word | (neighbor << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
            return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

        def route_top8(s, raws=None, bs=None):
            """Top-8 of sample s (call from one whole wave, after the router scores landed).

            Preserve all FP32 bits of (sigmoid + bias). Each round reduces the score
            first, then the expert ID only among exactly equal winners. Packing the
            ID into the score's low byte can change the top-k on real checkpoints.
            Candidate i of this lane is expert lane + 64 i.
            Returns (expert id, route weight = raw score / sum of
            the 8 raw scores * ROUTE_SCALE) of pick `lane` in score order, valid in
            lanes < TOP_K."""
            if const_expr(bs is None):
                bs = load_bias()
            if const_expr(raws is None):
                raws = getf_many(
                    [
                        (mb("scores"), s * N_EXPERTS + lane + i * 64)
                        for i in range(N_EXPERTS // 64)
                    ]
                )
                stamp("ug", bid, 7)
            ks = []
            for i in range_constexpr(N_EXPERTS // 64):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                ks.append(fx.Uint32(ok))
            ids = [lane + i * 64 for i in range(N_EXPERTS // 64)]
            # sort this lane's NC keys descending; each round then takes the wave max of
            # the lane heads and shifts the winning lane's list (0 is below every key)
            NC = N_EXPERTS // 64
            for a, b in (
                ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)) if NC == 4 else ((0, 1),)
            ):
                first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
                ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
                ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            mv = fx.Int32(0)  # lane k: the expert ID of pick k
            for k in range_constexpr(TOP_K):
                m = wave_umax_dpp(ks[0])
                winner = wave_umax_dpp((ks[0] == m).select(255 - ids[0], fx.Int32(0)))
                expert = 255 - winner
                hit = (ks[0] == m) & (ids[0] == expert)
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(NC)] + [ks[NC]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(NC)] + [ids[NC]]
                mv = write_lane_i32(expert, k, mv)
            e = mv
            src = (e % 64) * 4
            got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, N_EXPERTS // 64):
                raw = (e // 64 == i).select(got[i], raw)
            # ML4: renormalized softmax over the TOP_K picks = exp(l - max) / sum over picks
            valid = lane < TOP_K
            lg = valid.select(raw.bitcast(fx.Float32), fx.Float32(-1.0e30))
            mx = lg
            for off in (1, 2, 4):
                mx = _xred(mx, off, fmax)
            ex = valid.select(_exp(lg - mx), fx.Float32(0.0))
            tot = ex
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, ex * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials to every peer, then sum them in rank order.

            One wave owns each destination, allowing the peer stores to progress
            concurrently without retaining all peer pointers in every wave."""
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            if const_expr(W > 1):
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (tile // 2)
                            ri = (pair % (tile // 2)) * 2
                            put_bf(
                                peer_dst + region_base,
                                (rank * S + si) * HIDDEN + t * tile + ri,
                                [
                                    lds_ld(outs, si * tile + ri),
                                    lds_ld(outs, si * tile + ri + 1),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                if const_expr(callable(residual)):
                    r0, r1 = residual(s, row)
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                    got = []
                    if const_expr(not callable(residual)):
                        got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                else:
                    own = sym + region_base
                    specs = [
                        (own, ((src * S + s) * HIDDEN + row) // 2, 1)
                        for src in range(W)
                    ]
                    if const_expr(not callable(residual)):  # packed bf16 pair
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got[:W]]
                    got = got[W:]
                if const_expr(not callable(residual)):
                    r0, r1 = bf2_f32(got[0][0])
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, r0 + t0, r1 + t1)

        def dcp_merge_latent(s, t):
            if const_expr(D == 1):
                return True
            region_base = fx.Int64(SY["dcp"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_dcp_part_stride"]
            )
            summary_base = dcp_summary_index(rank, s, t, 0, S, N_UV)
            if wave < D:
                for batch in range_constexpr((DCP_SUMMARY_PAIRS + 63) // 64):
                    item = lane + batch * 64
                    if item < DCP_SUMMARY_PAIRS:
                        value = fx.Float32(0.0)
                        if item < KV_LORA // 2:
                            value = lds_ld(xs, item)
                        elif item == KV_LORA // 2:
                            value = lds_ld(misc, N_SPLIT)
                        else:
                            value = lds_ld(misc, N_SPLIT + 1)
                        put(peer_dst + region_base, summary_base + item, value, CM_SYS)
            gpu.barrier()
            owner = dcp_uv_owner(t, L, rank)
            if owner:
                own = sym + region_base
                if wave == 0:
                    src = fx.min(lane, D - 1)
                    src_base = dcp_summary_index(src, s, t, 0, S, N_UV)
                    values = poll(
                        [
                            (own, src_base + KV_LORA // 2, 1),
                            (own, src_base + KV_LORA // 2 + 1, 1),
                        ],
                        "one-as",
                    )
                    valid = lane < D
                    m = valid.select(values[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                    local_sum = valid.select(
                        values[1][0].bitcast(fx.Float32), fx.Float32(0.0)
                    )
                    m_all = wave_max(m)
                    z = local_sum * _exp(m - m_all)
                    den = wave_sum(z)
                    if valid:
                        lds_st(
                            misc,
                            lane,
                            z * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                        )
                gpu.barrier()
                if tid < KV_LORA // 2:
                    parts = poll(
                        [
                            (own, dcp_summary_index(src, s, t, tid, S, N_UV), 1)
                            for src in range(D)
                        ],
                        "one-as",
                    )
                    o0, o1 = fx.Float32(0.0), fx.Float32(0.0)
                    for src in range_constexpr(D):
                        a0, a1 = bf2_f32(parts[src][0])
                        weight = lds_ld(misc, src)
                        o0, o1 = o0 + a0 * weight, o1 + a1 * weight
                    lds_st(xs, tid, bf16_pair(o0, o1))
                gpu.barrier()
            return owner

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        # Read once: a param-table load inside a stage would retire every load
        # its wave has in flight (`vmcnt` is in order) and skew what it measures.
        tl_base = timeline_buf
        if const_expr(timeline and (with_indexer or separate_shared)):
            tl_base = index_arg(7)

        def stamp(name, t, which, lead=0):
            if const_expr(
                timeline
                and (not _TL_ONLY or name in _TL_ONLY)
                and (not _TL_COLS or which in _TL_COLS)
            ):
                # The lead's whole wave stores the same word under a uniform
                # branch, the address in SGPRs. Stamps still cost registers:
                # the full rope timeline build runs at 256 VGPRs with 80
                # spills (none without stamps), so `MONO_TL_ONLY` /
                # `MONO_TL_COLS` keep a measurement to the stages it is about.
                if _uniform(wave) == lead // 64:
                    bo.buffer_store(
                        mem_realtime(),
                        _rsrc(tl_base),
                        fx.Int32(0),
                        soffset_bytes=_uniform(
                            ((first[name] + t) * TL_COLS + which) * 8
                        ),
                    )

        def n_sel(count=S):
            """This lane's local MFMA B column; inactive columns duplicate the last one."""
            return fx.min(lane % 16, count - 1)

        # (``_SPLIT_PRE``) one load, one word per lane (``ld_row``): the rows'
        # positions, slots and index bounds, and per split round r of this
        # CTA its task's page (lane 48 + 2 r) and its row's page 0 (49 + 2 r).
        # Lanes past a table's end repeat its last entry.
        row_pre = None
        if const_expr(_SPLIT_PRE):
            pre_j, pre_k = lane // 16, lane % 16
            pre_row = fx.min(pre_k, fx.Int32(S - 1))
            pre_r = fx.min(pre_k // 2, fx.Int32((N_SPLIT_TASKS + G - 1) // G - 1))
            pre_s, pre_hg, pre_t = split_coords(start("split") + pre_r * G)
            pre_s = fx.min(pre_s, fx.Int32(S - 1))  # past the last task: unused
            pre_page = (pre_k % 2 == 0).select(
                fx.min(
                    pre_t * SPLIT_KEYS // index_page_size,
                    fx.Int32(index_block_table_stride - 1),
                ),
                fx.Int32(0),
            )
            pre_bt = (pre_s // index_request_width) * index_block_table_stride
            pre_ptr = (
                positions + pre_row * 8
                if const_expr(own_key_counts)
                else sparse_kv_indptr + fx.min(pre_k, fx.Int32(S)) * 4
            )
            pre_addr = (pre_j == 3).select(
                cur_pos + (pre_bt + pre_page) * 4,
                (pre_j == 2).select(
                    pre_ptr,
                    (pre_j == 1).select(
                        slot_mapping + pre_row * 8, positions + pre_row * 8
                    ),
                ),
            )
            row_pre = fx.Int32(global_load(pre_addr))

        # ================================================= 1. q_a / kv_a GEMV
        # 1 row group x 96 chunks: 8 waves split K, 12 chunks each (all prefetched)
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        QA_UNITS = QA_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wik, r_sik = _rsrc(index_arg(0)), _rsrc(index_arg(1))
            r_wiw = _rsrc(index_arg(2))
        IW_CPW = QA_NKC // WAVES
        N_QKV_T = N_QKV_A + N_QKV_IDX
        assert QKV_TILES in (1, (S + SAMPLE_TILE - 1) // SAMPLE_TILE)
        for tt in range(start("qkv_a"), N_QKV_T * QKV_TILES, G):
            tt = fx.Int32(tt)
            t = tt
            if const_expr(QKV_TILES > 1):
                t = tt % N_QKV_T
            stamp("qkv_a", tt, 0)

            def qkv_tile(sample_base):
                group_count = min(SAMPLE_TILE, S - sample_base)

                def u_qa(c):
                    kc = (wave * QA_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wqa,
                        r_sqa,
                        t,
                        kc,
                        QA_NKC,
                        HIDDEN,
                        128,
                        attention_b_word(n_sel(group_count) * HIDDEN + kc * 64),
                    )

                def ld_h(sks):
                    res = []
                    for s, k in sks:
                        w = fx.Vector(
                            bo.buffer_load(
                                r_h,
                                ((sample_base + s) * HIDDEN + k) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                def stage_h(issue):
                    """Stage this tile's RMSNorm(h) and issue() its weight units,
                    ahead of the staging when the h loads fit beside them."""
                    scales = None
                    if const_expr(S <= SAMPLE_TILE):
                        h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, group_count)
                        units = issue()
                        if const_expr(attention_ptpc):
                            scales = stage_x_rmsnorm_ptpc(
                                ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count
                            )
                        else:
                            stage_x_rmsnorm(
                                ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count
                            )
                    else:
                        if const_expr(attention_ptpc):
                            scales = stage_x_rmsnorm_ptpc(
                                ld_h, HIDDEN, g_in, count=group_count
                            )
                        else:
                            stage_x_rmsnorm(ld_h, HIDDEN, g_in, count=group_count)
                        units = issue()
                    gpu.barrier()
                    stamp("qkv_a", tt, 2)
                    return units, scales

                def kv_a_tile():
                    pre, qa_scales = stage_h(lambda: [u_qa(c) for c in range(QA_UNITS)])
                    acc = run_units(u_qa, QA_UNITS, QA_UNITS, pre)
                    if const_expr(attention_ptpc):
                        reduce_rows(
                            1,
                            acc,
                            emit_ptpc(QKV_A_TILE, t * QKV_A_TILE, r_sqa, qa_scales),
                            group_count,
                        )
                    else:
                        reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
                    stamp("qkv_a", tt, 3)
                    gpu.barrier()
                    if tid < group_count * QKV_A_TILE:
                        s = sample_base + tid // QKV_A_TILE
                        row = t * QKV_A_TILE + tid % QKV_A_TILE
                        v = lds_ld(outs, tid)
                        if row < Q_LORA:
                            put(mb("q_a"), s * Q_LORA + row, v)
                        else:
                            put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)

                def u_index_w(rg):
                    def make(c):
                        kc = wave * IW_CPW + c
                        return unit_bf16(
                            r_wiw,
                            rg,
                            kc,
                            QA_NKC,
                            (INDEX_BF16_OFF if const_expr(attention_ptpc) else 0)
                            + (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                        )

                    return make

                def u_index_k(rg):
                    def make(c):
                        kc = (wave * QA_UNITS + c) * 2
                        return unit_fp8x2(
                            r_wik,
                            r_sik,
                            rg,
                            kc,
                            QA_NKC,
                            HIDDEN,
                            (INDEX_BF16_OFF if const_expr(attention_ptpc) else 0)
                            + (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                        )

                    return make

                def put_index(acc, name, width, rg):
                    reduce_rows(1, acc, emit_out(INDEX_TILE), group_count)
                    gpu.barrier()
                    if tid < group_count * INDEX_TILE:
                        s = sample_base + tid // INDEX_TILE
                        row = rg * INDEX_TILE + tid % INDEX_TILE
                        put(mb(name), s * width + row, lds_ld(outs, tid))

                def index_tile(rg, units, cpw, name, width):
                    """One index key / head-weight tile as a task of its own."""
                    make = units(rg)
                    pre, _ = stage_h(lambda: [make(c) for c in range(cpw)])
                    put_index(run_units(make, cpw, cpw, pre), name, width, rg)

                if const_expr(N_QKV_IDX == 0):
                    kv_a_tile()
                    if const_expr(with_indexer):
                        if t < INDEX_DIM // QKV_A_TILE:
                            ik_acc = run_units(u_index_k(t), QA_UNITS, QA_UNITS)
                            iw_acc = [fx.Float32(0.0) for _ in range(4)]
                            if t < INDEX_HEADS // QKV_A_TILE:
                                iw_acc = run_units(u_index_w(t), IW_CPW, IW_CPW)
                            put_index(ik_acc, "index_k", INDEX_DIM, t)
                            if t < INDEX_HEADS // QKV_A_TILE:
                                put_index(iw_acc, "index_w", INDEX_HEADS, t)
                else:
                    if t < N_QKV_A:
                        kv_a_tile()
                    else:
                        if t < N_QKV_A + INDEX_DIM // QKV_A_TILE:
                            index_tile(
                                t - N_QKV_A, u_index_k, QA_UNITS, "index_k", INDEX_DIM
                            )
                        else:
                            index_tile(
                                t - (N_QKV_A + INDEX_DIM // QKV_A_TILE),
                                u_index_w,
                                IW_CPW,
                                "index_w",
                                INDEX_HEADS,
                            )

            for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                if const_expr(QKV_TILES > 1):
                    if tt // N_QKV_T == sample_base // SAMPLE_TILE:
                        qkv_tile(sample_base)
                else:
                    qkv_tile(sample_base)
            stamp("qkv_a", tt, 4)

        def ld_qa(sks):
            v = get2_many(
                [(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)]
            )
            return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

        # ===================== 2. ML4 (no q-LoRA): one CTA per sample publishes RMSNorm(x) * g_in (q_b input) and, with
        # the indexer, LayerNorm(that, eps 1e-6) * w + b (ML4's indexer q_input_norm; index_q input).  ``g_q`` points at
        # the fp32 ML4 parameter table (ML4_P_* offsets): indexer-q LayerNorm weight/bias, then the SSMax tables.
        r_ml4 = _rsrc(g_q)
        for s_norm in range(start("q_norm"), S, G):
            s_norm = fx.Int32(s_norm)
            stamp("q_norm", s_norm, 0)

            def ld_h_one(sks):
                res = []
                for _, k in sks:
                    w = fx.Vector(
                        bo.buffer_load(
                            r_h, (s_norm * HIDDEN + k) // 2, vec_width=2, dtype=T.i32
                        )
                    )
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            stage_x_rmsnorm(ld_h_one, HIDDEN, g_in, count=1)
            stamp("q_norm", s_norm, 2)
            gpu.barrier()
            normed = []
            for i in range_constexpr(HIDDEN // (4 * THREADS)):
                k = (tid + i * THREADS) * 4
                a0, a1 = bf2_f32(lds_ld(xs, k // 2).bitcast(fx.Int32))
                a2, a3 = bf2_f32(lds_ld(xs, k // 2 + 1).bitcast(fx.Int32))
                put_bf(mb("q_an"), s_norm * HIDDEN + k, [a0, a1, a2, a3])
                normed.append((k, [a0, a1, a2, a3]))
            if const_expr(with_indexer):
                tot = fx.Float32(0.0)
                for _, v in normed:
                    for a in v:
                        tot = tot + a
                mean = block_sum(tot) * (1.0 / HIDDEN)
                sq = fx.Float32(0.0)
                for _, v in normed:
                    for a in v:
                        sq = sq + (a - mean) * (a - mean)
                rstd = _rsq(block_sum(sq) * (1.0 / HIDDEN) + 1.0e-6)
                for k, v in normed:
                    put_bf(
                        mb("iq_an"),
                        s_norm * HIDDEN + k,
                        [
                            bf16_round(
                                (v[j] - mean) * rstd * ld_f32(r_ml4, ML4_P_IQ_W + k + j)
                                + ld_f32(r_ml4, ML4_P_IQ_B + k + j)
                            )
                            for j in range(4)
                        ],
                    )
            stamp("q_norm", s_norm, 4)

        # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        for t in range(start("cache"), CACHE_TASKS, G):
            stamp("cache", t, 0)
            # the task's rows: all of them, or row t alone (`cache_per_row`)
            rows = [fx.Int32(t)] if const_expr(CACHE_TASKS > 1) else list(range(S))
            n_rows = len(rows)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait; on gfx942
            # each row's position, slot and index cell too: loaded per row after
            # the stores below, every row would pay its own round trips
            g = ld_bf16(_rsrc(g_kv), tid)
            t_pe = tid % (PE_DIM // 2)
            if const_expr(GFX942):
                row_pos = [row_position(s) for s in rows]
                slots = [row_slot(s) for s in rows]

            def pos_of(i):
                return row_pos[i] if const_expr(GFX942) else row_position(rows[i])

            def slot_of(i):
                return slots[i] if const_expr(GFX942) else row_slot(rows[i])

            def writes_of(i):
                if const_expr(GFX942):
                    return slots[i] >= 0
                return row_writes_cache(rows[i])

            cs = [
                ld_rope(_rsrc(rope_cos), pos_of(i) * (PE_DIM // 2) + t_pe)
                for i in range(n_rows)
            ]
            sns = [
                ld_rope(_rsrc(rope_sin), pos_of(i) * (PE_DIM // 2) + t_pe)
                for i in range(n_rows)
            ]
            if const_expr(with_indexer and GFX942):
                # the index key's LayerNorm gamma / beta pair and each row's cell
                i_gb = fx.min(tid, INDEX_DIM // 2 - 1) * 2
                ik_gb = [
                    ld_f32(_rsrc(index_arg(a)), i_gb + j)
                    for a in (5, 6)
                    for j in (0, 1)
                ]
                if const_expr(index_fp8):
                    ik_cells = [index_page_cell(s, p) for s, p in zip(rows, row_pos)]
                else:
                    ik_cells = [index_page_slot(s, p) for s, p in zip(rows, row_pos)]
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (
                    mb("kv_a"),
                    rows[-1] * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
                mark=("cache", t),
            )
            # every sample's kv latent and k_pe pair in one poll, one block reduction
            vs = getf_many([(mb("kv_a"), s * (KV_LORA + PE_DIM) + tid) for s in rows])
            pes = get2_many(
                [
                    (
                        mb("kv_a"),
                        s * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2,
                    )
                    for s in rows
                ]
            )
            stamp("cache", t, 2)
            if const_expr(KV_SHARE):
                # ML4 KV-sharing layer: no latent of its own. Attend to the source layer's cache, whose rows for this
                # step were written earlier in the same step; never store into it.
                for i in range_constexpr(n_rows):
                    s, pos = rows[i], pos_of(i)
                    # A padded row (slot -1) has no source row: read slot 0 instead of
                    # wrapping the 32-bit buffer offset ~4 GiB past the cache (memory
                    # fault). Its kvnew is never read: patch_new_kv matches active rows only.
                    slot = writes_of(i).select(slot_of(i), fx.Int32(0))
                    if const_expr(cache_fp8):
                        # fp8 implies the ATOM layout (asserted above): one byte per value
                        kvn = ld_fp8(r_kv, slot * QK_DIM + tid)
                    elif const_expr(use_atom_kv_cache):
                        kvn = ld_bf16(r_kv, slot * QK_DIM + tid)
                    else:
                        kvn = ld_bf16(r_kv, pos * KV_LORA + tid)
                    put(mb("kvnew"), s * KV_LORA + tid, kvn)
                    if tid < PE_DIM // 2:
                        if const_expr(cache_fp8):
                            p0, p1 = ld_fp8_pair(
                                r_pe, (slot * QK_DIM + KV_LORA) // 2 + tid
                            )
                        elif const_expr(use_atom_kv_cache):
                            p0 = ld_bf16(r_pe, slot * QK_DIM + KV_LORA + tid * 2)
                            p1 = ld_bf16(r_pe, slot * QK_DIM + KV_LORA + tid * 2 + 1)
                        else:
                            p0 = ld_bf16(r_pe, pos * PE_DIM + tid * 2)
                            p1 = ld_bf16(r_pe, pos * PE_DIM + tid * 2 + 1)
                        put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)
            else:
                ssq = block_sums([v * v for v in vs])
                for i in range_constexpr(n_rows):
                    s, pos, slot, active = rows[i], pos_of(i), slot_of(i), writes_of(i)
                    value = vs[i] * _rsq(ssq[i] * (1.0 / KV_LORA) + EPS) * g
                    if const_expr(cache_fp8):
                        # saturate like vLLM's fp8 cache writer (and torch): the
                        # conversion turns |x| > FP8_MAX into NaN
                        value = _fp8_sat(bf16_round(value))
                        kvn, _ = _fp8_roundtrip(value, value)
                    else:
                        kvn = bf16_round(value)
                    if const_expr(use_atom_kv_cache):
                        if active:
                            if const_expr(cache_fp8):
                                pair = fx.Int32(
                                    rocdl.cvt_pk_fp8_f32(
                                        T.i32, kvn, _xshfl(kvn, 1), fx.Int32(0), False
                                    )
                                ) & fx.Int32(0xFFFF)
                                upper_pair = bpermute_i32(
                                    fp8_kv_upper_pair_lane(lane) * 4, pair
                                )
                                if lane % 4 == 0:
                                    bo.buffer_store(
                                        pair | (upper_pair << 16),
                                        r_kv,
                                        slot * (QK_DIM // 4) + tid // 4,
                                    )
                            else:
                                bo.buffer_store(
                                    kvn.to(fx.BFloat16), r_kv, slot * QK_DIM + tid
                                )
                    else:
                        bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                    put(mb("kvnew"), s * KV_LORA + tid, kvn)
                    if tid < PE_DIM // 2:
                        x0, x1 = pes[i]
                        c, sn = cs[i], sns[i]
                        p0, p1 = x0 * c - x1 * sn, x0 * sn + x1 * c
                        if const_expr(cache_fp8):
                            p0 = _fp8_sat(bf16_round(p0))
                            p1 = _fp8_sat(bf16_round(p1))
                            p0, p1 = _fp8_roundtrip(p0, p1)
                        else:
                            p0, p1 = bf16_round(p0), bf16_round(p1)
                        if const_expr(use_atom_kv_cache):
                            if active:
                                if const_expr(cache_fp8):
                                    pair = fx.Int32(
                                        rocdl.cvt_pk_fp8_f32(
                                            T.i32, p0, p1, fx.Int32(0), False
                                        )
                                    ) & fx.Int32(0xFFFF)
                                    upper_pair = bpermute_i32(
                                        fp8_pe_upper_pair_lane(lane) * 4, pair
                                    )
                                    if lane % 2 == 0:
                                        bo.buffer_store(
                                            pair | (upper_pair << 16),
                                            r_pe,
                                            slot * (QK_DIM // 4)
                                            + KV_LORA // 4
                                            + tid // 2,
                                        )
                                else:
                                    pe_offset = slot * QK_DIM + KV_LORA + tid * 2
                                    bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                                    bo.buffer_store(
                                        p1.to(fx.BFloat16), r_pe, pe_offset + 1
                                    )
                        else:
                            pe_offset = pos * PE_DIM + tid * 2
                            bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                            bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                        put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)

            if const_expr(with_indexer):
                # Index keys use LayerNorm (not RMSNorm), interleaved RoPE on
                # the first 64 dimensions, and the build's index cache format.
                # Hadamard is omitted because applying the same orthogonal
                # transform to Q and K leaves their dot products unchanged.
                if const_expr(not GFX942):
                    r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                r_index_cache = _rsrc(indices)

                def ik_gamma(i0, j):
                    return ik_gb[j] if const_expr(GFX942) else ld_f32(r_gik, i0 + j)

                def ik_beta(i0, j):
                    return ik_gb[2 + j] if const_expr(GFX942) else ld_f32(r_bik, i0 + j)

                def ik_cell(i):
                    if const_expr(GFX942):
                        return ik_cells[i]
                    s = rows[i]
                    if const_expr(index_fp8):
                        return index_page_cell(s, row_position(s))
                    return index_page_slot(s, row_position(s))

                for i in range_constexpr(n_rows):
                    s = rows[i]
                    # Same gate the MLA kv/pe writes above use: a pad row has
                    # slot_mapping -1 and row_position 0, so an ungated store
                    # would land on row 0 of whatever page this request's block
                    # table happens to point at and clobber a live request's
                    # position-0 index key. The mailbox publishes below stay
                    # ungated -- index_score polls every row.
                    active = writes_of(i)
                    ik = getf(mb("index_k"), s * INDEX_DIM + fx.min(tid, INDEX_DIM - 1))
                    live = tid < INDEX_DIM
                    iv = live.select(ik, fx.Float32(0.0))
                    mean = block_sum(iv) * (1.0 / INDEX_DIM)
                    centered = live.select(ik - mean, fx.Float32(0.0))
                    rstd = _rsq(
                        block_sum(centered * centered) * (1.0 / INDEX_DIM) + 1.0e-6
                    )
                    if tid < INDEX_DIM // 2:
                        i0 = tid * 2
                        k0, k1 = get2(mb("index_k"), s * INDEX_DIM + i0)
                        v0 = (k0 - mean) * rstd * ik_gamma(i0, 0) + ik_beta(i0, 0)
                        v1 = (k1 - mean) * rstd * ik_gamma(i0, 1) + ik_beta(i0, 1)
                        if const_expr(INDEX_ROPE):
                            if tid < PE_DIM // 2:
                                c, sn = cs[i], sns[i]
                                v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
                        if const_expr(index_fp8):
                            # ATOM's plane: the production writer rounds the
                            # roped key back through bf16, takes one ue8m0 scale
                            # over the row's own 128 values, and stores E4M3
                            # bytes preshuffled into MFMA 16x16 tiles with the
                            # fp32 scales in a separate per-page segment.
                            v0, v1 = bf16_round(v0), bf16_round(v1)
                            amax = fmax(
                                wave_max(fmax(fmath.absf(v0), fmath.absf(v1))),
                                fx.Float32(1.0e-4),
                            )
                            # exp2(ceil(log2(amax / INDEX_FP8_MAX))) as an exact
                            # statement about the exponent field: INDEX_FP8_MAX
                            # = 1.75 * 2**d, so amax = m * 2**a with m <= 1.75
                            # gives 2**(a - d), m > 1.75 gives 2**(a - d + 1).
                            # No divide, no log2, so no 1-ulp slip across a
                            # power of two -- which would halve a row.
                            bits = amax.bitcast(fx.Int32)
                            exponent = ((bits >> 23) & fx.Int32(0xFF)) - INDEX_FP8_EXP
                            exponent = exponent + (
                                (bits & fx.Int32(0x7FFFFF)) > fx.Int32(0x600000)
                            ).select(fx.Int32(1), fx.Int32(0))
                            exponent = fx.max(
                                fx.min(exponent, fx.Int32(254)), fx.Int32(1)
                            )
                            qs = (exponent << 23).bitcast(fx.Float32)
                            inv = ((fx.Int32(254) - exponent) << 23).bitcast(fx.Float32)
                            q0 = fmin(fmax(v0 * inv, -INDEX_FP8_MAX), INDEX_FP8_MAX)
                            q1 = fmin(fmax(v1 * inv, -INDEX_FP8_MAX), INDEX_FP8_MAX)
                            page, row = ik_cell(i)
                            pair = fx.Int32(
                                rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)
                            ) & fx.Int32(0xFFFF)
                            upper_pair = bpermute_i32(
                                fp8_pe_upper_pair_lane(lane) * 4, pair
                            )
                            if active:
                                if lane % 2 == 0:
                                    bo.buffer_store(
                                        pair | (upper_pair << 16),
                                        r_index_cache,
                                        index_fp8_data_dword(
                                            page, row, i0, index_page_size
                                        ),
                                    )
                                if tid == 0:
                                    bo.buffer_store(
                                        qs,
                                        r_index_cache,
                                        index_fp8_scale_dword(
                                            page, row, index_page_size
                                        ),
                                    )
                            # Consumer: the index_score stage's own-row branch
                            # (`safe_key >= first_position`, below, which selects
                            # the mailbox over a cache read precisely for the
                            # key positions this very launch just wrote and that
                            # r_index_cache is not yet guaranteed to reflect this
                            # step). Publish the dequantized value, not raw v0/v1,
                            # so those keys score at the same E4M3 precision a
                            # cache read would give them -- production never
                            # scores a key that bypassed quantization, and the
                            # BF16 arm has no such gap because it has no
                            # quantization step to bypass. qs is a power of two,
                            # so bf16(q * qs) is bit-identical to the cache read
                            # times qs, not merely close to it.
                            d0, d1 = _fp8_roundtrip(q0, q1)
                            put_bf(
                                mb("index_k_new"),
                                s * INDEX_DIM + i0,
                                [d0 * qs, d1 * qs],
                            )
                        else:
                            if active:
                                bo.buffer_store(
                                    fx.Vector.from_elements([v0, v1], fx.Float32).to(
                                        fx.BFloat16
                                    ),
                                    r_index_cache,
                                    ik_cell(i) * INDEX_DIM + i0,
                                )
                            put_bf(mb("index_k_new"), s * INDEX_DIM + i0, [v0, v1])
                    gpu.barrier()
                    if tid == 0:
                        put(mb("index_ready"), s, fx.Int32(1))
            stamp("cache", t, 4)

        # =============================================== 4. normalized q_a -> q_b (+RoPE)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = HIDDEN // 64
        QB_UNITS = QB_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)

            def u_qb(c):
                kc = (wave * QB_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wqb,
                    r_sqb,
                    t,
                    kc,
                    QB_NKC,
                    HIDDEN,
                    128,
                    attention_b_word(n_sel(X_TILE) * HIDDEN + kc * 64),
                )

            pre = [u_qb(c) for c in range(QB_UNITS)]
            hint_wait(
                S,
                lambda s: (mb("q_an"), (s * HIDDEN + HIDDEN - 2) // 2),
                mark=("q_b", t),
            )
            if const_expr(attention_ptpc):
                qb_scales = stage_x_pairs_ptpc(
                    "q_an", S, Q_LORA, lambda k: k, retain_bf16=with_indexer
                )
                stamp("q_b", t, 2)
                gpu.barrier()
                acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
                reduce_rows(1, acc, emit_ptpc(Q_B_TILE, t * Q_B_TILE, r_sqb, qb_scales))
            else:
                gemv_x_tiles("q_an", pre, Q_B_TILE, mark=("q_b", t))
            stamp("q_b", t, 3)
            gpu.barrier()
            head = t // QB_PER_HEAD
            hoff = (t % QB_PER_HEAD) * Q_B_TILE
            if const_expr(Q_PE_DIM == 0):
                if (hoff == 0) & (tid < S * PE_DIM // 4):
                    zs = tid // (PE_DIM // 4)
                    put_bf(
                        mb("q_pe"),
                        (zs * H + head) * PE_DIM + (tid % (PE_DIM // 4)) * 4,
                        [fx.Float32(0.0)] * 4,
                    )
            if hoff < NOPE_DIM:
                if tid < S * Q_B_TILE // 4:
                    s = tid // (Q_B_TILE // 4)
                    r = (tid % (Q_B_TILE // 4)) * 4
                    ss = ml4_ssmax(row_position(s), ML4_P_SS_S, ML4_P_SS_B, head)
                    put_bf(
                        mb("q_nope"),
                        (s * H + head) * NOPE_DIM + hoff + r,
                        [
                            bf16_round(
                                bf16_round(lds_ld(outs, s * Q_B_TILE + r + j)) * ss
                            )
                            for j in range(4)
                        ],
                    )
            else:
                if tid < S * Q_B_TILE // 2:
                    s = tid // (Q_B_TILE // 2)
                    pr = tid % (Q_B_TILE // 2)
                    i = hoff - NOPE_DIM + pr * 2
                    x0 = lds_ld(outs, s * Q_B_TILE + pr * 2)
                    x1 = lds_ld(outs, s * Q_B_TILE + pr * 2 + 1)
                    c = ld_rope(
                        _rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + i // 2
                    )
                    sn = ld_rope(
                        _rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + i // 2
                    )
                    ss = ml4_ssmax(row_position(s), ML4_P_SS_S, ML4_P_SS_B, head)
                    x0, x1 = bf16_round(x0), bf16_round(x1)
                    p0, p1 = bf16_round(x0 * c - x1 * sn), bf16_round(x0 * sn + x1 * c)
                    put_bf(
                        mb("q_pe"),
                        (s * H + head) * PE_DIM + i,
                        [bf16_round(p0 * ss), bf16_round(p1 * ss)],
                    )
            stamp("q_b", t, 4)

            if const_expr(with_indexer and not SHARD_INDEX_Q):

                def qb_index_q():
                    # Reuse this CTA's normalized q_lora tile for one index-query
                    # row tile.  Complementary CTAs compute the remaining half below.
                    gpu.barrier()
                    iq_t = t
                    stamp("index_q", iq_t, 0)

                    # ML4: the indexer query reads LayerNorm(normed x), not this CTA's staged q_b input -- re-stage it.
                    def u_index_q(c):
                        kc = (wave * QB_UNITS + c) * 2
                        return unit_fp8x2(
                            r_wiq,
                            r_siq,
                            iq_t,
                            kc,
                            QB_NKC,
                            HIDDEN,
                            (n_sel(X_TILE) * HIDDEN + kc * 64) // 2,
                        )

                    gemv_x_tiles(
                        "iq_an",
                        lambda: [u_index_q(c) for c in range(QB_UNITS)],
                        INDEX_TILE,
                    )
                    stamp("index_q", iq_t, 3)
                    gpu.barrier()
                    if tid < S * INDEX_TILE // 4:
                        s = tid // (INDEX_TILE // 4)
                        r = (tid % (INDEX_TILE // 4)) * 4
                        put_bf(
                            mb("index_q"),
                            s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                            [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                        )
                    stamp("index_q", iq_t, 4)

                if const_expr(_DENSE_SHORT):
                    if any_long_row():
                        qb_index_q()
                else:
                    qb_index_q()

        if const_expr(with_indexer):
            # The 128 CTAs without q_b work produce the other 128 index-query
            # tiles concurrently.  They reload q_a, but remove one full GEMV
            # from the q_b CTAs' serialized critical path.
            N_INDEX_Q_EXTRA = (
                INDEX_Q_ROWS // INDEX_TILE // CP_RANKS
                if SHARD_INDEX_Q
                else INDEX_Q_ROWS // INDEX_TILE - N_QB
            )
            n_index_q_extra = (
                any_long_row().select(fx.Int32(N_INDEX_Q_EXTRA), fx.Int32(0))
                if const_expr(_DENSE_SHORT)
                else N_INDEX_Q_EXTRA
            )
            for tt in range(start("index_q"), n_index_q_extra, G):
                tt = fx.Int32(tt)
                iq_t = (
                    cp_local_rank() * N_INDEX_Q_EXTRA + tt
                    if const_expr(SHARD_INDEX_Q)
                    else N_QB + tt
                )
                iq_task = tt if const_expr(SHARD_INDEX_Q) else iq_t
                stamp("index_q", iq_task, 0)

                def u_index_q_extra(c):
                    kc = (wave * QB_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wiq,
                        r_siq,
                        iq_t,
                        kc,
                        QB_NKC,
                        HIDDEN,
                        (n_sel(X_TILE) * HIDDEN + kc * 64) // 2,
                    )

                pre = [u_index_q_extra(c) for c in range(QB_UNITS)]
                hint_wait(
                    S,
                    lambda s: (mb("iq_an"), (s * HIDDEN + HIDDEN - 2) // 2),
                    mark=("index_q", iq_task),
                )
                gemv_x_tiles("iq_an", pre, INDEX_TILE, mark=("index_q", iq_task))
                stamp("index_q", iq_task, 3)
                gpu.barrier()
                if const_expr(SHARD_INDEX_Q):
                    if (wave < CP_RANKS) & (lane < S * INDEX_TILE // 4):
                        s = lane // (INDEX_TILE // 4)
                        r = (lane % (INDEX_TILE // 4)) * 4
                        put_bf(
                            cp_buffer(
                                "index_cp_q",
                                cp_peer(wave) if const_expr(CP_RANKS < W) else peer_dst,
                            ),
                            s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                            [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                            CM_SYS,
                        )
                else:
                    if tid < S * INDEX_TILE // 4:
                        s = tid // (INDEX_TILE // 4)
                        r = (tid % (INDEX_TILE // 4)) * 4
                        put_bf(
                            mb("index_q"),
                            s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                            [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                        )
                stamp("index_q", iq_task, 4)

        # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
        # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per wave
        r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
        UK_NKC = NOPE_DIM // 64
        for t in range(start("uk"), N_UK, G):
            t = fx.Int32(t)
            stamp("uk", t, 0)
            head = t // UK_PER_HEAD

            def u_uk(c):
                return unit_attention(
                    r_wuk,
                    r_suk,
                    t * WAVES + wave,
                    c,
                    UK_NKC,
                    NOPE_DIM,
                    64,
                    attention_b_word(n_sel() * NOPE_DIM + c * 64),
                    coef=(
                        (
                            lambda: pick(
                                [
                                    uk_scales[
                                        sample * ((NOPE_DIM + 127) // 128) + c // 2
                                    ]
                                    for sample in range(S)
                                ],
                                n_sel(),
                            )
                        )
                        if const_expr(attention_ptpc)
                        else None
                    ),
                )

            pre = [u_uk(c) for c in range(UK_NKC)]
            hint_wait(
                NOPE_DIM // Q_B_TILE,
                lambda k: (
                    mb("q_nope"),
                    ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1,
                ),
                mark=("uk", t),
            )
            if const_expr(attention_ptpc):
                uk_scales = stage_x_pairs_ptpc(
                    "q_nope",
                    S,
                    NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                    group=128,
                )
            else:
                stage_x_pairs(
                    "q_nope",
                    S * NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                )
            stamp("uk", t, 2)
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            stamp("uk", t, 3)
            gpu.barrier()
            if tid < S * UK_TILE // 4:
                k = tid * 4
                s = k // UK_TILE
                r0 = k % UK_TILE
                vals = []
                if const_expr(attention_ptpc):
                    ptpc_scale = ld_f32(r_suk, 0)
                for j in range_constexpr(4):
                    r = r0 + j
                    ww = r // 16
                    value = lds_ld(
                        red, (ww * 64 + s + 16 * ((r % 16) // 4)) * 4 + r % 4
                    )
                    if const_expr(attention_ptpc):
                        value = bf16_round(value * ptpc_scale)
                    vals.append(value)
                put_bf(
                    mb("q_lat"),
                    (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0,
                    vals,
                )
            stamp("uk", t, 4)

        # ====================== 4b. fused sparse index score + exact top-2048
        if const_expr(with_indexer):
            r_index_cache = _rsrc(indices)
            N_INDEX_SPLIT = INDEX_CAPACITY // INDEX_KEYS_PER_TASK

            def use_long_index(s):
                if const_expr(indexer_cp):
                    return row_position(s) + 1 > LOCAL_INDEX_CAPACITY
                return fx.Boolean(True)

            def use_short_index(s):
                # A short compiled capacity cannot cross the selector boundary.
                # Preserve the original branch-free short path with CP off.
                if const_expr(not TILED_INDEXER):
                    return fx.Boolean(True)
                return row_position(s) + 1 <= LOCAL_INDEX_CAPACITY

            def index_shard(s):
                bound = row_position(s) + 1
                if const_expr(indexer_cp):
                    stride = (
                        (bound + CP_RANKS * INDEX_KEYS_PER_TASK - 1)
                        // (CP_RANKS * INDEX_KEYS_PER_TASK)
                    ) * INDEX_KEYS_PER_TASK
                    begin = cp_local_rank() * stride
                    return begin, fx.max(fx.Int32(0), fx.min(stride, bound - begin))
                return fx.Int32(0), bound

            def load_index_q8(head, k):
                words = fx.Vector.from_elements(
                    [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)],
                    fx.Float32,
                )
                return words.bitcast(fx.BFloat16)

            for tt in range(start("index_score"), S * N_INDEX_SPLIT, G):
                tt = fx.Int32(tt)
                s = tt // N_INDEX_SPLIT
                tile = tt % N_INDEX_SPLIT
                stamp("index_score", tt, 0)
                bound = row_position(s) + 1
                request_start = index_request_start(s)
                first_position = row_position(request_start)
                shard_begin, shard_size = index_shard(s)
                if (tile * INDEX_KEYS_PER_TASK < shard_size) & long_row(s):
                    get(mb("index_ready"), s)
                    if tid < INDEX_HEADS:
                        lds_st(keys, tid, get(mb("index_w"), s * INDEX_HEADS + tid))
                    # Every key group reuses the same 32x128 query.  Stage and RoPE
                    # it once per scoring CTA instead of polling and rotating it in
                    # each of the four key-group wave pairs.
                    # Peer reads benefit from independent outstanding polls.
                    # Keep the original one-pair staging order when CP is off.
                    q_batch = (INDEX_Q_ROWS // 2) // THREADS if indexer_cp else 1
                    for batch in range_constexpr(
                        0, (INDEX_Q_ROWS // 2) // THREADS, q_batch
                    ):
                        q_pairs = [tid + (batch + b) * THREADS for b in range(q_batch)]
                        q_source = (
                            cp_buffer("index_cp_q", sym)
                            if const_expr(SHARD_INDEX_Q)
                            else mb("index_q")
                        )
                        q_words = poll(
                            [
                                (q_source, s * INDEX_Q_ROWS // 2 + pair, 1)
                                for pair in q_pairs
                            ],
                            scope="system" if SHARD_INDEX_Q else "agent",
                        )
                        for q_pair, q_word in zip(q_pairs, q_words):
                            q_elem = q_pair * 2
                            kq = q_elem % INDEX_DIM
                            q0, q1 = bf2_f32(q_word[0])
                            iss = ml4_ssmax(
                                row_position(s),
                                ML4_P_ISS_S,
                                ML4_P_ISS_B,
                                q_elem // INDEX_DIM,
                            )
                            q0, q1 = bf16_round(q0 * iss), bf16_round(q1 * iss)
                            if const_expr(INDEX_ROPE):
                                if kq < PE_DIM:
                                    c = ld_rope(
                                        _rsrc(rope_cos),
                                        row_position(s) * (PE_DIM // 2) + kq // 2,
                                    )
                                    sn = ld_rope(
                                        _rsrc(rope_sin),
                                        row_position(s) * (PE_DIM // 2) + kq // 2,
                                    )
                                    q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                            lds_st(xs, q_pair, bf16_pair(q0, q1))
                    gpu.barrier()
                    key_group = wave // 2
                    head_group = wave % 2
                    local_key = tile * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                    key_pos = shard_begin + local_key
                    safe_key = fx.min(key_pos, bound - 1)
                    head = head_group * 16 + lane % 16
                    if const_expr(index_fp8):
                        index_page, index_row = index_page_cell(s, safe_key)
                        kv_scale = ld_f32(
                            r_index_cache,
                            index_fp8_scale_dword(
                                index_page, index_row, index_page_size
                            ),
                        )
                    score_frag = fx.Vector.filled(4, 0.0, fx.Float32)
                    for k32 in range_constexpr(INDEX_DIM // 32):
                        k = k32 * 32 + (lane // 16) * 8
                        qv = load_index_q8(head, k)
                        if const_expr(index_fp8):
                            # Eight preshuffled E4M3 bytes; the row's eight
                            # columns k..k+7 share one 16-column tile, so they
                            # are contiguous.  E4M3 -> bf16 is exact, and the
                            # row scale multiplies the finished score instead of
                            # every key element.
                            fp8_words = fx.Vector(
                                bo.buffer_load(
                                    r_index_cache,
                                    index_fp8_data_dword(
                                        index_page, index_row, k, index_page_size
                                    ),
                                    vec_width=2,
                                    dtype=T.i32,
                                )
                            )
                            kv = _fp8_to_bf16x8(fp8_words[0], fp8_words[1])
                        else:
                            kv = fx.Vector(
                                bo.buffer_load(
                                    r_index_cache,
                                    (index_page_slot(s, safe_key) * INDEX_DIM + k) // 2,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            ).bitcast(fx.BFloat16)
                        if safe_key >= first_position:
                            sn = request_start + safe_key - first_position
                            kv_pairs = get_bf2_many(
                                [
                                    (mb("index_k_new"), sn * INDEX_DIM + k + j * 2)
                                    for j in range(4)
                                ]
                            )
                            kv_values = []
                            for pair in kv_pairs:
                                kv_values += list(pair)
                            kv = fx.Vector.from_elements(kv_values, fx.Float32).to(
                                fx.BFloat16
                            )
                        score_frag = _mfma_bf16(qv, kv, score_frag)
                    partial = fx.Float32(0.0)
                    for e in range_constexpr(4):
                        index_head = head_group * 16 + (lane // 16) * 4 + e
                        weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                        partial = (
                            partial + fmax(score_frag[e], fx.Float32(0.0)) * weight
                        )
                    if const_expr(index_fp8):
                        # Rows this launch wrote came from the mailbox already
                        # dequantized; only cache reads carry a scale.  The key
                        # is lane % 16, which the two reductions below preserve,
                        # so one factor per lane is correct either side of them.
                        partial = partial * (safe_key >= first_position).select(
                            fx.Float32(1.0), kv_scale
                        )
                    partial = _xred(partial, 16, lambda a, b: a + b)
                    partial = _xred(partial, 32, lambda a, b: a + b)
                    if lane < 16:
                        lds_st(red, wave * 16 + lane, partial)
                    gpu.barrier()
                    if (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                        score = lds_ld(red, wave * 16 + lane) + lds_ld(
                            red, (wave + 1) * 16 + lane
                        )
                        if const_expr(indexer_cp):
                            if bound <= LOCAL_INDEX_CAPACITY:
                                lds_st(red, 128 + key_group * 16 + lane, score)
                            else:
                                put(
                                    mb("index_scores"),
                                    s * INDEX_CAPACITY + local_key,
                                    score,
                                )
                        else:
                            put(
                                mb("index_scores"),
                                s * INDEX_CAPACITY + local_key,
                                score,
                            )
                    if const_expr(indexer_cp):
                        if bound <= LOCAL_INDEX_CAPACITY:
                            # One wave per peer, one complete score exchange.
                            gpu.barrier()
                            out_key = shard_begin + tile * INDEX_KEYS_PER_TASK + lane
                            if (wave < CP_RANKS) & (out_key < bound):
                                put(
                                    cp_buffer(
                                        "index_cp_scores",
                                        cp_peer(wave)
                                        if const_expr(CP_RANKS < W)
                                        else peer_dst,
                                    ),
                                    s * LOCAL_INDEX_CAPACITY + out_key,
                                    lds_ld(red, 128 + lane),
                                    CM_SYS,
                                )
                stamp("index_score", tt, 4)

            if const_expr(TILED_INDEXER):
                # Context-parallel exact radix selection. Scoring owns a
                # contiguous, 64-key-aligned shard of the *live* context on each
                # rank. Histograms and compaction use bounded 4K-key tasks;
                # neither LDS nor the compiled loop body scales with context.
                # Each rank completes its local stage before entering the next
                # one. All producers are resident, including rank-0 coordinators.
                def cp_get(name, i):
                    return poll([(cp_buffer(name, sym), i, 1)], scope="system")[0][0]

                def cp_threshold(digit, s):
                    values = poll(
                        [
                            (
                                cp_buffer("index_cp_threshold", sym),
                                (digit * S + s) * 2,
                                2,
                            )
                        ],
                        scope="system",
                    )[0]
                    return fx.Uint32(_uniform(values[0])), _uniform(values[1])

                def score_key(s, i):
                    bits = get(mb("index_scores"), s * INDEX_CAPACITY + i)
                    return fx.Uint32(
                        (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)
                    )

                def cp_scan(value):
                    """Exclusive scan in ascending thread order; all threads participate."""
                    inclusive = fx.coop.warp_inclusive_scan(
                        value, fx.ReductionOp.ADD, width=64
                    )
                    if lane == 63:
                        lds_st(keys, 256 + wave, inclusive)
                    gpu.barrier()
                    before = fx.Int32(0)
                    total = fx.Int32(0)
                    for w in range_constexpr(WAVES):
                        count = lds_ld(keys, 256 + w)
                        before = before + (wave > w).select(count, fx.Int32(0))
                        total = total + count
                    offset = before + inclusive - value
                    gpu.barrier()
                    return offset, total

                def cp_select_digit(shift, prefix, remain):
                    digit = 255 - fx.min(tid, 255)
                    count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                    above, _ = cp_scan(count)
                    hit = (tid < 256) & (above < remain) & (above + count >= remain)
                    if hit:
                        lds_st(
                            keys, 256, fx.Int32(prefix | (fx.Uint32(digit) << shift))
                        )
                        lds_st(keys, 257, remain - above)
                    gpu.barrier()
                    result = fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)
                    gpu.barrier()
                    return result

                cp_rank = cp_local_rank() if const_expr(indexer_cp) else fx.Int32(0)
                owner = cp_peer(fx.Int32(0))
                for digit in range_constexpr(INDEX_RADIX_DIGITS):
                    shift = 24 - digit * 8
                    mask = (0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF
                    for tt in range(start(f"index_hist_{digit}"), S * CP_PARTS, G):
                        tt = fx.Int32(tt)
                        s, part = tt // CP_PARTS, tt % CP_PARTS
                        stamp(f"index_hist_{digit}", tt, 0)
                        if use_long_index(s):
                            _, size = index_shard(s)
                            prefix = fx.Uint32(0)
                            if const_expr(digit > 0):
                                prefix, _ = cp_threshold(digit - 1, s)
                            if tid < INDEX_RADIX_BINS:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()
                            for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                                i = part * INDEX_CP_TILE + batch * THREADS + tid
                                if i < size:
                                    key = score_key(s, i)
                                    if (key & fx.Uint32(mask)) == prefix:
                                        bucket = fx.Int32(
                                            (key >> shift) & fx.Uint32(255)
                                        )
                                        fx.atomic_add(
                                            keys + bucket,
                                            fx.Int32(1),
                                            syncscope="workgroup",
                                        )
                            gpu.barrier()
                            if tid < INDEX_RADIX_BINS:
                                put(
                                    mb("index_cp_hist"),
                                    (digit * S * CP_PARTS + tt) * 256 + tid,
                                    lds_ld(keys, tid),
                                )
                            gpu.barrier()
                        stamp(f"index_hist_{digit}", tt, 4)

                    for s in range(start(f"index_prefix_{digit}"), S, G):
                        s = fx.Int32(s)
                        stamp(f"index_prefix_{digit}", s, 0)
                        if use_long_index(s):
                            if tid < INDEX_RADIX_BINS:
                                count = fx.Int32(0)
                                for part in range(CP_PARTS):
                                    count = count + get(
                                        mb("index_cp_hist"),
                                        ((digit * S + s) * CP_PARTS + part) * 256 + tid,
                                    )
                                put(
                                    cp_buffer("index_cp_hist", owner),
                                    ((cp_rank * INDEX_RADIX_DIGITS + digit) * S + s)
                                    * 256
                                    + tid,
                                    count,
                                    CM_SYS,
                                )
                            gpu.barrier()
                            if cp_rank == 0:
                                if tid < INDEX_RADIX_BINS:
                                    count = fx.Int32(0)
                                    for src in range_constexpr(CP_RANKS):
                                        count = count + cp_get(
                                            "index_cp_hist",
                                            ((src * INDEX_RADIX_DIGITS + digit) * S + s)
                                            * 256
                                            + tid,
                                        )
                                    lds_st(keys, tid, count)
                                gpu.barrier()
                                prefix = fx.Uint32(0)
                                remain = fx.min(fx.Int32(topk), row_position(s) + 1)
                                if const_expr(digit > 0):
                                    prefix, remain = cp_threshold(digit - 1, s)
                                prefix, remain = cp_select_digit(shift, prefix, remain)
                                if tid == 0:
                                    for dst in range_constexpr(CP_RANKS):
                                        address = cp_buffer(
                                            "index_cp_threshold", cp_peer(fx.Int32(dst))
                                        )
                                        put(
                                            address,
                                            (digit * S + s) * 2,
                                            fx.Int32(prefix),
                                            CM_SYS,
                                        )
                                        put(
                                            address,
                                            (digit * S + s) * 2 + 1,
                                            remain,
                                            CM_SYS,
                                        )
                            gpu.barrier()
                        stamp(f"index_prefix_{digit}", s, 4)

                # Count strict winners and threshold ties separately. Prefixes
                # in rank/partition/token order preserve the original selector's
                # output: ascending token IDs above threshold, then earliest ties.
                for tt in range(start("index_count"), S * CP_PARTS, G):
                    tt = fx.Int32(tt)
                    s, part = tt // CP_PARTS, tt % CP_PARTS
                    stamp("index_count", tt, 0)
                    if use_long_index(s):
                        _, size = index_shard(s)
                        threshold, _ = cp_threshold(3, s)
                        n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                        for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                            i = part * INDEX_CP_TILE + batch * THREADS + tid
                            if i < size:
                                key = score_key(s, i)
                                n_gt = n_gt + (key > threshold).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                                n_eq = n_eq + (key == threshold).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                        _, total_gt = cp_scan(n_gt)
                        _, total_eq = cp_scan(n_eq)
                        if tid == 0:
                            put(mb("index_cp_counts"), tt * 2, total_gt)
                            put(mb("index_cp_counts"), tt * 2 + 1, total_eq)
                        gpu.barrier()
                    stamp("index_count", tt, 4)

                for s in range(start("index_offsets"), S, G):
                    s = fx.Int32(s)
                    stamp("index_offsets", s, 0)
                    if use_long_index(s):
                        total_gt, total_eq = fx.Int32(0), fx.Int32(0)
                        for batch in range_constexpr(
                            (CP_PARTS + THREADS - 1) // THREADS
                        ):
                            part = batch * THREADS + tid
                            n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                            if part < CP_PARTS:
                                n_gt = get(
                                    mb("index_cp_counts"), (s * CP_PARTS + part) * 2
                                )
                                n_eq = get(
                                    mb("index_cp_counts"), (s * CP_PARTS + part) * 2 + 1
                                )
                            off_gt, batch_gt = cp_scan(n_gt)
                            off_eq, batch_eq = cp_scan(n_eq)
                            if part < CP_PARTS:
                                put(
                                    mb("index_cp_offsets"),
                                    (s * CP_PARTS + part) * 2,
                                    total_gt + off_gt,
                                )
                                put(
                                    mb("index_cp_offsets"),
                                    (s * CP_PARTS + part) * 2 + 1,
                                    total_eq + off_eq,
                                )
                            total_gt = total_gt + batch_gt
                            total_eq = total_eq + batch_eq
                        if tid == 0:
                            put(
                                cp_buffer("index_cp_counts", owner),
                                (cp_rank * S + s) * 2,
                                total_gt,
                                CM_SYS,
                            )
                            put(
                                cp_buffer("index_cp_counts", owner),
                                (cp_rank * S + s) * 2 + 1,
                                total_eq,
                                CM_SYS,
                            )
                        gpu.barrier()
                        if cp_rank == 0:
                            before_gt, before_eq = fx.Int32(0), fx.Int32(0)
                            for src in range_constexpr(CP_RANKS):
                                n_gt = cp_get("index_cp_counts", (src * S + s) * 2)
                                n_eq = cp_get("index_cp_counts", (src * S + s) * 2 + 1)
                                if tid == 0:
                                    address = cp_buffer(
                                        "index_cp_offsets", cp_peer(fx.Int32(src))
                                    )
                                    put(address, s * 2, before_gt, CM_SYS)
                                    put(address, s * 2 + 1, before_eq, CM_SYS)
                                before_gt = before_gt + n_gt
                                before_eq = before_eq + n_eq
                        gpu.barrier()
                    stamp("index_offsets", s, 4)

                for tt in range(start("index_emit"), S * CP_PARTS, G):
                    tt = fx.Int32(tt)
                    s, part = tt // CP_PARTS, tt % CP_PARTS
                    stamp("index_emit", tt, 0)
                    if use_long_index(s):
                        begin, size = index_shard(s)
                        threshold, need_eq = cp_threshold(3, s)
                        before_gt = cp_get("index_cp_offsets", s * 2) + get(
                            mb("index_cp_offsets"), tt * 2
                        )
                        before_eq = cp_get("index_cp_offsets", s * 2 + 1) + get(
                            mb("index_cp_offsets"), tt * 2 + 1
                        )
                        for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                            i = part * INDEX_CP_TILE + batch * THREADS + tid
                            is_gt, is_eq = fx.Int32(0), fx.Int32(0)
                            if i < size:
                                key = score_key(s, i)
                                is_gt = (key > threshold).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                                is_eq = (key == threshold).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                            off_gt, count_gt = cp_scan(is_gt)
                            off_eq, count_eq = cp_scan(is_eq)
                            eq_index = before_eq + off_eq
                            selected = (is_gt != 0) | (
                                (is_eq != 0) & (eq_index < need_eq)
                            )
                            # Same combined offset the local selector emits with,
                            # so both selectors lay a row out ASCENDING BY KEY
                            # INDEX and stay comparable bit for bit.  The set is
                            # untouched: ``min(eq_index, need_eq)`` is the number
                            # of taken ties below this key, which for a tie is
                            # its own rank among the taken ones.
                            slot = before_gt + off_gt + fx.min(eq_index, need_eq)
                            if selected:
                                for dst in range_constexpr(CP_RANKS):
                                    put(
                                        cp_buffer(
                                            "index_cp_indices", cp_peer(fx.Int32(dst))
                                        ),
                                        s * topk + slot,
                                        begin + i,
                                        CM_SYS,
                                    )
                            before_gt = before_gt + count_gt
                            before_eq = before_eq + count_eq
                        gpu.barrier()
                    stamp("index_emit", tt, 4)

                for s in range(start("index_select"), S, G):
                    s = fx.Int32(s)
                    stamp("index_select", s, 0)
                    if use_long_index(s):
                        bound = row_position(s) + 1
                        for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                            j = batch * THREADS + tid
                            value = fx.Int32(0)
                            if j < fx.min(fx.Int32(topk), bound):
                                value = cp_get("index_cp_indices", s * topk + j)
                            bo.buffer_store(
                                value,
                                _rsrc(mb("indices")),
                                s * topk + j,
                                cache_modifier=CM_DEV,
                            )
                        fx.memory_fence(
                            ordering=fx.AtomicOrdering.Release, syncscope="agent"
                        )
                        gpu.barrier()
                        if tid == 0:
                            put(mb("indices_ready"), s, fx.Int32(1))
                    stamp("index_select", s, 4)

            if const_expr(not TILED_INDEXER or indexer_cp):
                # ---------------------------------------- exact top-2048, as a tree
                # One task ranks a FIXED ``SEG`` keys in LDS.  Level 0 ranks a
                # segment of the score plane; level l > 0 re-ranks ``FAN`` of level
                # l - 1's lists laid end to end, which is exact because a global
                # winner is never outside its own list's top-``topk``.  Capacity is
                # therefore ``SEG * FAN ** DEPTH`` for ``DEPTH`` unrolled levels and
                # the LDS staging never grows -- at ``SELECT_CAPACITY <= SEG``,
                # which includes every CP build, ``DEPTH`` is 0 and this is exactly
                # the single flat select it replaces.
                SEG = LOCAL_INDEX_CAPACITY
                FAN = INDEX_FAN
                DEPTH = INDEX_DEPTH
                SEL_ITEMS = (SEG + THREADS - 1) // THREADS
                # Row stride of the level-0 source.  A CP build reads the
                # peer-exchanged plane, which is one leaf wide; everything else
                # reads the scorer's own plane, which is the whole context.
                SCORE_STRIDE = LOCAL_INDEX_CAPACITY if indexer_cp else INDEX_CAPACITY

                for level in range_constexpr(DEPTH + 1):
                    SLOTS = index_lists(SELECT_CAPACITY, level, topk)
                    SEL_STAGE = index_select_stage(level, DEPTH)
                    # A level's region holds a score plane then an identity plane,
                    # list j's entry e of each at ``j * topk + e``.  Both are tagged
                    # pairs: the score poll is what makes the entry visible, and the
                    # identity poll (one batch, already landed) is what carries the
                    # original key index up the tree.
                    BELOW = (
                        index_lists(SELECT_CAPACITY, level - 1, topk) if level else 0
                    )
                    src = mb(f"isel{level - 1}") if level else mb("index_scores")
                    src_ident = (src + fx.Int64(S * BELOW * topk * 8)) if level else src
                    dst = mb(f"isel{level}") if level < DEPTH else src
                    dst_ident = dst + fx.Int64(S * SLOTS * topk * 8)
                    for tt in range(start(SEL_STAGE), S * SLOTS, G):
                        tt = fx.Int32(tt)
                        if const_expr(SLOTS == 1):
                            s, j = tt, fx.Int32(0)
                        else:
                            s = tt // SLOTS
                            j = tt - s * SLOTS
                        if use_short_index(s):
                            bound = row_position(s) + 1
                            # Lists this token runs at this level, and whether this
                            # level's answer is the token's: both traced on
                            # ``bound``, so a context below the build's maximum ends
                            # the tree early without a second compiled path.
                            if (j < tree_count(bound, level, SEG, FAN)) & long_row(s):
                                stamp(SEL_STAGE, tt, 0)
                                n_in = tree_inputs(bound, level, j, SEG, FAN, topk)
                                at_final = (
                                    fx.Boolean(True)
                                    if const_expr(level == DEPTH)
                                    else tree_final(bound, level, SEG, FAN)
                                )

                                def entry_pair(i, j=j, s=s, BELOW=BELOW):
                                    """Pair index of merged value `i` in the level below."""
                                    sub = i // topk
                                    return (s * BELOW + j * FAN + sub) * topk + (
                                        i - sub * topk
                                    )

                                def select_digit(shift, prefix, remain):
                                    digit = 255 - fx.min(tid, 255)
                                    count = (tid < 256).select(
                                        lds_ld(keys, digit), fx.Int32(0)
                                    )
                                    inclusive = fx.coop.warp_inclusive_scan(
                                        count, fx.ReductionOp.ADD, width=64
                                    )
                                    wave_total = read_lane_i32(inclusive, 63)
                                    if (wave < 4) & (lane == 63):
                                        lds_st(keys, 256 + wave, wave_total)
                                    gpu.barrier()
                                    before_wave = fx.Int32(0)
                                    for w in range_constexpr(4):
                                        before_wave = before_wave + (wave > w).select(
                                            lds_ld(keys, 256 + w), fx.Int32(0)
                                        )
                                    above = before_wave + inclusive - count
                                    hit = (
                                        (tid < 256)
                                        & (above < remain)
                                        & (above + count >= remain)
                                    )
                                    gpu.barrier()
                                    if hit:
                                        lds_st(
                                            keys,
                                            256,
                                            fx.Int32(
                                                prefix | (fx.Uint32(digit) << shift)
                                            ),
                                        )
                                        lds_st(keys, 257, remain - above)
                                    gpu.barrier()
                                    result = (
                                        fx.Uint32(lds_ld(keys, 256)),
                                        lds_ld(keys, 257),
                                    )
                                    # The following scan reuses these words for wave totals.
                                    # All waves must read the final threshold before any
                                    # wave overwrites it, or compaction can exceed topk.
                                    gpu.barrier()
                                    return result

                                # Transform scores to monotonic integer keys and build the high
                                # byte histogram in the same pass, avoiding one extra 4-K LDS
                                # scan before the remaining radix digits.  A list below the top
                                # already carries order keys, so only level 0 transforms.
                                if tid < 256:
                                    lds_st(keys, tid, fx.Int32(0))
                                gpu.barrier()

                                def stage_index_key(i, bits):
                                    if i < n_in:
                                        key = (bits >= 0).select(
                                            bits ^ fx.Int32(-(2**31)), ~bits
                                        )
                                        # The selector CTA no longer needs the large GEMV staging
                                        # region, so reuse it for the 4-K radix keys instead of
                                        # increasing the monokernel's LDS allocation.
                                        lds_st(xs, i, key.bitcast(fx.Float32))
                                        digit = fx.Int32(
                                            (fx.Uint32(key) >> 24) & fx.Uint32(255)
                                        )
                                        fx.atomic_add(
                                            keys + digit,
                                            fx.Int32(1),
                                            syncscope="workgroup",
                                        )

                                if const_expr(indexer_cp):
                                    # Issue several independent peer-mailbox reads together.
                                    # Tail lanes poll a valid key but never enter the histogram.
                                    score_batches = SEL_ITEMS
                                    for batch in range_constexpr(0, score_batches, 4):
                                        ids = [
                                            tid + b * THREADS
                                            for b in range(
                                                batch, min(batch + 4, score_batches)
                                            )
                                        ]
                                        values = poll(
                                            [
                                                (
                                                    cp_buffer("index_cp_scores", sym),
                                                    s * SCORE_STRIDE
                                                    + fx.min(i, n_in - 1),
                                                    1,
                                                )
                                                for i in ids
                                            ],
                                            scope="system",
                                        )
                                        for i, value in zip(ids, values):
                                            stage_index_key(i, value[0])
                                else:
                                    for batch in range_constexpr(SEL_ITEMS):
                                        i = tid + batch * THREADS
                                        # The source read is a tagged-pair POLL: it must stay
                                        # INSIDE this guard.  A lane past ``n_in`` would spin
                                        # forever on an entry its producer never wrote.
                                        if i < n_in:
                                            if const_expr(level == 0):
                                                bits = getf(
                                                    src, s * SCORE_STRIDE + j * SEG + i
                                                ).bitcast(fx.Int32)
                                                key = (bits >= 0).select(
                                                    bits ^ fx.Int32(-(2**31)), ~bits
                                                )
                                            else:
                                                # A list below the top already carries order keys.
                                                key = get(src, entry_pair(i))
                                            # The selector CTA no longer needs the large GEMV
                                            # staging region, so reuse it for the 4-K radix keys
                                            # instead of growing the monokernel's LDS allocation.
                                            lds_st(xs, i, key.bitcast(fx.Float32))
                                            digit = fx.Int32(
                                                (fx.Uint32(key) >> 24) & fx.Uint32(255)
                                            )
                                            fx.atomic_add(
                                                keys + digit,
                                                fx.Int32(1),
                                                syncscope="workgroup",
                                            )
                                gpu.barrier()
                                prefix, remain = select_digit(
                                    24, fx.Uint32(0), fx.min(fx.Int32(topk), n_in)
                                )
                                prefix_mask = 255 << 24
                                for shift in (16, 8, 0):
                                    if tid < 256:
                                        lds_st(keys, tid, fx.Int32(0))
                                    gpu.barrier()
                                    for batch in range_constexpr(SEL_ITEMS):
                                        i = tid + batch * THREADS
                                        if i < n_in:
                                            key = fx.Uint32(
                                                lds_ld(xs, i).bitcast(fx.Int32)
                                            )
                                            if (key & fx.Uint32(prefix_mask)) == prefix:
                                                digit = fx.Int32(
                                                    (key >> shift) & fx.Uint32(255)
                                                )
                                                fx.atomic_add(
                                                    keys + digit,
                                                    fx.Int32(1),
                                                    syncscope="workgroup",
                                                )
                                    gpu.barrier()
                                    prefix, remain = select_digit(shift, prefix, remain)
                                    prefix_mask |= 255 << shift

                                threshold = prefix
                                r_index_out = _rsrc(mb("indices"))
                                items = SEL_ITEMS
                                item_indices = [
                                    tid * items + d for d in range_constexpr(items)
                                ]
                                item_bits = [
                                    lds_ld(xs, fx.min(i, n_in - 1)).bitcast(fx.Int32)
                                    for i in item_indices
                                ]
                                item_keys = [fx.Uint32(b) for b in item_bits]
                                gt = [
                                    (i < n_in) & (key > threshold)
                                    for i, key in zip(item_indices, item_keys)
                                ]
                                eq = [
                                    (i < n_in) & (key == threshold)
                                    for i, key in zip(item_indices, item_keys)
                                ]
                                # What each winner stands for.  Level 0's values are
                                # the score plane itself, so a position is its own
                                # identity; a merge level's are a list below, which
                                # carried theirs in the identity plane beside the
                                # scores.  ONE batched poll, not a load inside each
                                # winner's branch, so the whole row costs one round
                                # trip rather than ``items``.
                                if const_expr(level == 0):
                                    idents = [
                                        fx.Int32(j * SEG + i) for i in item_indices
                                    ]
                                else:
                                    idents = [
                                        v[0]
                                        for v in poll(
                                            [
                                                (
                                                    src_ident,
                                                    entry_pair(fx.min(i, n_in - 1)),
                                                    1,
                                                )
                                                for i in item_indices
                                            ]
                                        )
                                    ]

                                def scan_flags(flags):
                                    """Thread-major exclusive offsets for eight flags without extra LDS.

                                    The radix histogram already reserves keys[256:264] for wave
                                    totals, so the scan can reuse those words and keep S=8 at the
                                    96-KiB static-LDS boundary.
                                    """
                                    local = fx.Int32(0)
                                    local_offsets = []
                                    for flag in flags:
                                        local_offsets.append(local)
                                        local = local + flag.select(
                                            fx.Int32(1), fx.Int32(0)
                                        )
                                    inclusive = fx.coop.warp_inclusive_scan(
                                        local, fx.ReductionOp.ADD, width=64
                                    )
                                    wave_total = read_lane_i32(inclusive, 63)
                                    if lane == 63:
                                        lds_st(keys, 256 + wave, wave_total)
                                    gpu.barrier()
                                    before_wave = fx.Int32(0)
                                    total = fx.Int32(0)
                                    for w in range_constexpr(WAVES):
                                        wave_count = lds_ld(keys, 256 + w)
                                        before_wave = before_wave + (wave > w).select(
                                            wave_count, fx.Int32(0)
                                        )
                                        total = total + wave_count
                                    thread_base = before_wave + inclusive - local
                                    return [
                                        thread_base + off for off in local_offsets
                                    ], total

                                gt_offsets, out_gt = scan_flags(gt)
                                gpu.barrier()
                                eq_offsets, _ = scan_flags(eq)
                                gpu.barrier()
                                need_eq = fx.min(fx.Int32(topk), n_in) - out_gt
                                # ONE combined offset instead of a strict-greater block
                                # followed by a tie block: the winners come out dense and
                                # ASCENDING BY KEY INDEX, which is what a merge level above
                                # must be able to assume and what makes ``entry_pair``'s
                                # flat addressing exact.  The set is unchanged -- same
                                # threshold, same ``need_eq``, same lowest-index tie rule --
                                # and the attention gather never depended on the order
                                # (``split_keys`` reads ``indices`` through
                                # ``index_page_slot``, a permutation-invariant gather).
                                for d in range_constexpr(items):
                                    win = gt[d] | (eq[d] & (eq_offsets[d] < need_eq))
                                    pos = gt_offsets[d] + fx.min(eq_offsets[d], need_eq)
                                    if win:
                                        if const_expr(level == DEPTH):
                                            bo.buffer_store(
                                                idents[d],
                                                r_index_out,
                                                s * topk + pos,
                                                cache_modifier=CM_DEV,
                                            )
                                        else:
                                            if at_final:
                                                bo.buffer_store(
                                                    idents[d],
                                                    r_index_out,
                                                    s * topk + pos,
                                                    cache_modifier=CM_DEV,
                                                )
                                            else:
                                                put(
                                                    dst,
                                                    (s * SLOTS + j) * topk + pos,
                                                    item_bits[d],
                                                )
                                                put(
                                                    dst_ident,
                                                    (s * SLOTS + j) * topk + pos,
                                                    idents[d],
                                                )
                                if at_final:
                                    # An explicit zero sentinel past the valid count, which
                                    # every reader of ``indices`` relies on.
                                    for batch in range_constexpr(
                                        (topk + THREADS - 1) // THREADS
                                    ):
                                        d = tid + batch * THREADS
                                        if d >= bound:
                                            bo.buffer_store(
                                                fx.Int32(0),
                                                r_index_out,
                                                s * topk + d,
                                                cache_modifier=CM_DEV,
                                            )
                                # Every lane contributed compact indices.  Make all of those
                                # device-memory stores visible before lane 0 publishes the one
                                # readiness tag consumed by the attention CTAs.
                                fx.memory_fence(
                                    ordering=fx.AtomicOrdering.Release,
                                    syncscope="agent",
                                )
                                gpu.barrier()
                                if at_final:
                                    if tid == 0:
                                        put(mb("indices_ready"), s, fx.Int32(1))
                                stamp(SEL_STAGE, tt, 4)

        # ================================== 5. sparse MLA split: 32 keys x 8 heads
        r_kv = _rsrc(kv_cache)
        r_pe = _rsrc(pe_cache)
        r_idx = _rsrc(indices)
        KPW = SPLIT_KEYS // WAVES

        def page_key_slot(k_pos, nkeys, pages):
            """``index_page_slot`` of a dense row's key k_pos -- key 0's past
            nkeys -- from its split's (page, page 0) ``split_pre_pick``."""
            page_t, page_0 = pages
            return (k_pos < nkeys).select(
                page_t * index_page_size + k_pos % index_page_size,
                page_0 * index_page_size,
            )

        def selected_key(s, k):
            """Key position k of row s's indexer top-k."""
            return fx.Int32(
                bo.buffer_load(
                    _rsrc(mb("indices")),
                    s * topk + k,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )

        def split_keys(t, s, rows=None, pages=None):
            """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS.
            ``rows``: ``split_rows(s)`` if already loaded; ``pages``: its
            ``split_pre_pick`` pages (``_SPLIT_PRE``)."""
            nkeys, sparse, use_indices, index_base = (
                split_rows(s) if const_expr(rows is None) else rows
            )
            if const_expr(with_indexer):
                if use_indices:
                    if wave == 0:
                        if lane == 0:
                            get(mb("indices_ready"), s)
                        # Only wave 0 consumes the compact index payload; its
                        # divergent ready poll reconverges before this acquire.
                        fx.memory_fence(
                            ordering=fx.AtomicOrdering.Acquire, syncscope="agent"
                        )
            if wave == 0:
                if lane < SPLIT_KEYS:
                    k_pos = t * SPLIT_KEYS + lane
                    k_cl = (k_pos < nkeys).select(k_pos, 0)
                    if const_expr(with_indexer and pages is not None):
                        k_slot = page_key_slot(k_pos, nkeys, pages)
                        if use_indices:
                            k_slot = index_page_slot(s, selected_key(s, k_cl))
                        lds_st(attn_keys, lane, k_slot)
                    elif const_expr(with_indexer):
                        idx = k_cl
                        if use_indices:
                            idx = selected_key(s, k_cl)
                        lds_st(attn_keys, lane, index_page_slot(s, idx))
                    elif const_expr(shared_indices and pages is not None):
                        k_slot = sparse.select(
                            page_key_slot(k_pos, nkeys, pages), fx.Int32(0)
                        )
                        if use_indices:
                            k_slot = index_page_slot(
                                s,
                                fx.Int32(
                                    bo.buffer_load(
                                        r_idx, s * topk + k_cl, vec_width=1, dtype=T.i32
                                    )
                                ),
                            )
                        lds_st(attn_keys, lane, k_slot)
                    elif const_expr(shared_indices):
                        idx = fx.Int32(0)
                        if sparse:
                            pos_k = fx.Int32(
                                bo.buffer_load(
                                    r_idx, s * topk + k_cl, vec_width=1, dtype=T.i32
                                )
                            )
                            if const_expr(_DENSE_SHORT):
                                pos_k = use_indices.select(pos_k, k_cl)
                            idx = index_page_slot(s, pos_k)
                        lds_st(attn_keys, lane, idx)
                    elif const_expr(use_atom_kv_cache):
                        idx = fx.Int32(0)
                        if sparse:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    r_idx, index_base + k_cl, vec_width=1, dtype=T.i32
                                )
                            )
                        lds_st(attn_keys, lane, idx)
                    else:
                        idx = fx.Int32(
                            bo.buffer_load(
                                r_idx, index_base + k_cl, vec_width=1, dtype=T.i32
                            )
                        )
                        lds_st(attn_keys, lane, sparse.select(idx, k_cl))
            return nkeys, sparse

        def gather_old_kv_batched(krows):
            """``gather_old_kv`` with every row load issued before the first wait.
            Every lane loads and stores k_pe -- past its lanes, the same bytes
            to the same LDS words as lane % PE_LANES: a divergent store would
            draw its loads into the branch, after the latent's wait."""
            PE_LANES = PE_DIM // 8 if const_expr(cache_fp8) else PE_DIM // 2
            pe_lane = lane % PE_LANES
            if const_expr(cache_fp8):
                rows = [krows[jj] * (QK_DIM // 4) for jj in range(KPW)]
                kvs = [
                    fx.Vector(
                        bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32)
                    )
                    for row in rows
                ]
                pes = [
                    fx.Vector(
                        bo.buffer_load(
                            r_pe,
                            row + KV_LORA // 4 + pe_lane * 2,
                            vec_width=2,
                            dtype=T.i32,
                        )
                    )
                    for row in rows
                ]
            else:
                kv_row_words = (
                    QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                )
                kvs = [
                    fx.Vector(
                        bo.buffer_load(
                            r_kv,
                            kr * kv_row_words + lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for kr in krows
                ]
                pes = [
                    ld_f32(
                        r_pe,
                        kr * (QK_DIM // 2) + KV_LORA // 2 + pe_lane
                        if const_expr(use_atom_kv_cache)
                        else kr * (PE_DIM // 2) + pe_lane,
                    )
                    for kr in krows
                ]
            if const_expr(cache_fp8):
                for jj in range_constexpr(KPW):
                    fx.ptr_store(
                        _fp8_to_bf16x8(kvs[jj][0], kvs[jj][1]).bitcast(fx.Float32),
                        ktile + ((wave * KPW + jj) * KS + lane * 4),
                    )
                for jj in range_constexpr(KPW):
                    fx.ptr_store(
                        _fp8_to_bf16x8(pes[jj][0], pes[jj][1]).bitcast(fx.Float32),
                        petile + ((wave * KPW + jj) * PS + pe_lane * 4),
                    )
            else:
                for jj in range_constexpr(KPW):
                    fx.ptr_store(
                        kvs[jj].bitcast(fx.Float32),
                        ktile + ((wave * KPW + jj) * KS + lane * 4),
                    )
                for jj in range_constexpr(KPW):
                    lds_st(petile, (wave * KPW + jj) * PS + pe_lane, pes[jj])

        def gather_old_kv():
            """Each wave copies its 8 keys' KV latent (1 KB) + k_pe (128 B) cache rows
            into the LDS tiles (rows of this launch are patched in by patch_new_kv)."""
            krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
            if const_expr(_GATHER_BATCH):
                gather_old_kv_batched(krows)
            else:
                gather_old_kv_serial(krows)

        def gather_old_kv_serial(krows):
            """``gather_old_kv`` key by key: each key's divergent k_pe load splits
            the block, so its loads wait one after another."""
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                if const_expr(cache_fp8):
                    row = krows[jj] * (QK_DIM // 4)
                    raw = fx.Vector(
                        bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32)
                    )
                    fx.ptr_store(
                        _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                        ktile + (j * KS + lane * 4),
                    )
                    if lane < PE_DIM // 8:
                        pe_raw = fx.Vector(
                            bo.buffer_load(
                                r_pe,
                                row + KV_LORA // 4 + lane * 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        fx.ptr_store(
                            _fp8_to_bf16x8(pe_raw[0], pe_raw[1]).bitcast(fx.Float32),
                            petile + (j * PS + lane * 4),
                        )
                else:
                    kv_row_words = (
                        QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                    )
                    kv8 = fx.Vector(
                        bo.buffer_load(
                            r_kv,
                            krows[jj] * kv_row_words + lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                    if lane < PE_DIM // 2:
                        pe_row = (
                            krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                            if const_expr(use_atom_kv_cache)
                            else krows[jj] * (PE_DIM // 2) + lane
                        )
                        lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

        def patch_new_kv():
            """Rows appended by this launch come from the cache task's kvnew / penew pairs."""
            if const_expr(_SPLIT_PRE):
                new_slots = [row_slot(new_s, pre=True) for new_s in range(S)]
                new_active = [sl >= 0 for sl in new_slots]
            elif const_expr(use_atom_kv_cache):
                new_active = [row_writes_cache(new_s) for new_s in range(S)]
                new_slots = [row_slot(new_s) for new_s in range(S)]

            def key_row(jj):
                """(the launch row of the wave's key jj, whether it is one)."""
                kr = lds_ld(attn_keys, wave * KPW + jj)
                if const_expr(use_atom_kv_cache):
                    sn = fx.Int32(-1)
                    for new_s in range_constexpr(S):
                        match = new_active[new_s] & (kr == new_slots[new_s])
                        sn = match.select(fx.Int32(new_s), sn)
                    return sn, sn >= 0
                is_new = kr >= pos0
                return kr - pos0, is_new

            if const_expr(_PATCH_BATCH or GFX942):
                # the launch row of each of the wave's keys, -1 for an older row
                sns = []
                for jj in range_constexpr(KPW):
                    sn, is_new = key_row(jj)
                    if const_expr(not use_atom_kv_cache):
                        sn = is_new.select(sn, fx.Int32(-1))
                    sns.append(sn)
            if const_expr(_PATCH_BATCH):
                patch_new_kv_batched(sns)
            elif const_expr(GFX942):
                # every key's row before the first patch, so the key reads and
                # matches overlap
                patch_new_kv_serial(lambda jj: (sns[jj], sns[jj] >= 0))
            else:
                patch_new_kv_serial(key_row)

        def patch_new_kv_batched(sns):
            """``patch_new_kv`` in one round trip: a wave holding rows of this
            launch polls the kvnew / penew pairs of all its keys (or, with fewer
            launch rows than keys, of all the rows it holds) together, in
            batches of two all in flight at once; a source the wave does not
            need polls a needed one's pairs instead."""
            sn_any = sns[0]
            for jj in range_constexpr(1, KPW):
                sn_any = fx.max(sn_any, sns[jj])
            if sn_any >= 0:
                pe_lane = lane % (PE_DIM // 2)
                BY_ROW = S < KPW
                if const_expr(BY_ROW):
                    psrcs = []
                    for r in range_constexpr(S):
                        held = sns[0] == r
                        for jj in range_constexpr(1, KPW):
                            held = held | (sns[jj] == r)
                        psrcs.append(held.select(fx.Int32(r), sn_any))
                else:
                    psrcs = [
                        (sns[jj] >= 0).select(sns[jj], sn_any) for jj in range(KPW)
                    ]
                pspecs = []
                for psn in psrcs:
                    pspecs += [
                        (mb("kvnew"), psn * KV_LORA + lane * 8 + m * 2, 2)
                        for m in range(4)
                    ]
                    pspecs.append((mb("penew"), psn * PE_DIM + pe_lane * 2, 2))
                pgroups = [pspecs[g : g + 10] for g in range(0, len(pspecs), 10)]
                pfirst = [poll_issue(pg) for pg in pgroups]
                pgot = []
                for pg, pf in zip(pgroups, pfirst):
                    pgot += poll(pg, first=pf)
                for jj in range_constexpr(KPW):
                    if sns[jj] >= 0:
                        if const_expr(BY_ROW):
                            psel = pgot[0:5]
                            for r in range_constexpr(1, S):
                                psel = [
                                    [
                                        (sns[jj] == r).select(nw, ow)
                                        for nw, ow in zip(nv, ov)
                                    ]
                                    for nv, ov in zip(pgot[r * 5 : r * 5 + 5], psel)
                                ]
                        else:
                            psel = pgot[jj * 5 : jj * 5 + 5]
                        pkv = [
                            bf16_pair(
                                v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)
                            )
                            for v in psel[0:4]
                        ]
                        fx.ptr_store(
                            fx.Vector.from_elements(pkv, fx.Float32),
                            ktile + ((wave * KPW + jj) * KS + lane * 4),
                        )
                        lds_st(
                            petile,
                            (wave * KPW + jj) * PS + pe_lane,
                            bf16_pair(
                                psel[4][0].bitcast(fx.Float32),
                                psel[4][1].bitcast(fx.Float32),
                            ),
                        )

        def patch_new_kv_serial(key_row):
            """``patch_new_kv`` key by key, kvnew then penew."""
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                sn, is_new = key_row(jj)
                if is_new:
                    kvp = get2_many(
                        [
                            (mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2)
                            for m in range(4)
                        ]
                    )
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                    fx.ptr_store(
                        fx.Vector.from_elements(w, fx.Float32),
                        ktile + (j * KS + lane * 4),
                    )
                    if lane < PE_DIM // 2:
                        a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                        lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

        # (``SPLIT_MERGE``: the merged head groups, at the build scope)
        SPLIT_QH = H if SPLIT_MERGE else QHEADS

        def row_nkeys(s):
            """``split_rows``'s key count of row s."""
            if const_expr(use_atom_kv_cache):
                begin, end = row_index_bounds(s)
                return end - begin
            return fx.min(pos0 + s + 1, fx.Int32(topk))

        def live_splits(nkeys):
            """Splits holding keys, at least one (``_SKIP_EMPTY_SPLITS``)."""
            return fx.max((nkeys + (SPLIT_KEYS - 1)) // SPLIT_KEYS, fx.Int32(1))

        def split_task(tt, s, hg0, t, rows=None, pages=None):
            nkeys, sparse = split_keys(t, s, rows, pages)
            gpu.barrier()
            stamp("split", tt, 7)
            gather_old_kv()  # before waiting for q: these rows are from earlier launches
            if const_expr(True):
                N_PE_T = PE_DIM // Q_B_TILE
                hint_wait(
                    N_UK + H * N_PE_T + 1,
                    lambda k: (
                        (k < N_UK).select(
                            fx.Int64(SC["q_lat"]),
                            (k < N_UK + H * N_PE_T).select(
                                fx.Int64(SC["q_pe"]), fx.Int64(SC["penew"])
                            ),
                        )
                        + scratch,
                        (k < N_UK).select(
                            (s * H + k // UK_PER_HEAD) * KV_LORA
                            + (k % UK_PER_HEAD) * UK_TILE
                            + UK_TILE
                            - 1,
                            (k < N_UK + H * N_PE_T).select(
                                (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                                + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                + Q_B_TILE
                                - 1,
                                s * PE_DIM + PE_DIM - 1,
                            ),
                        ),
                    ),
                    mark=("split", tt),
                )
            # q -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64.
            # A task only ever reads its own head group's WAVES rows (``hn`` and
            # ``h`` are both ``head_group * WAVES + ...``), so stage only those:
            # with N_HEAD_GROUPS = 2 the old all-H staging polled and stored
            # twice the q words it used, in every one of the S * 2 * N_SPLIT
            # tasks. Bit-identical -- the skipped LDS rows are never read.
            NQ = SPLIT_QH * KV_LORA // 4 // THREADS
            t_pe = fx.min(tid, SPLIT_QH * PE_DIM // 4 - 1)
            qh0 = hg0 * SPLIT_QH if const_expr(SPLIT_QH != H) else fx.Int32(0)
            qv = poll(
                [
                    (
                        mb("q_lat"),
                        ((s * H + qh0) * KV_LORA + (tid + i * THREADS) * 4) // 2,
                        2,
                    )
                    for i in range(NQ)
                ]
                + [(mb("q_pe"), ((s * H + qh0) * PE_DIM + t_pe * 4) // 2, 2)]
            )
            for i in range_constexpr(NQ):
                w4 = tid + i * THREADS
                qw = (qh0 + w4 // (KV_LORA // 4)) * QS + (w4 % (KV_LORA // 4)) * 2
                lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
            if tid < SPLIT_QH * PE_DIM // 4:
                hh = qh0 + tid // (PE_DIM // 4)
                qw = hh * QS + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                lds_st(xs, qw, qv[NQ][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[NQ][1].bitcast(fx.Float32))
            patch_new_kv()
            if const_expr(True):
                stamp("split", tt, 2)
            gpu.barrier()
            stamp("split", tt, 5)
            # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
            # (four row groups, two K halves); the batch-8 32-key tile uses two
            # row groups and four waves per K half.  All waves subsequently own
            # one attention head for softmax and P@V.
            for hgi in range_constexpr(SPLIT_GROUPS):
                head_group = (hg0 + hgi) if const_expr(SPLIT_MERGE) else hg0
                h = head_group * WAVES + wave
                if const_expr(hgi):
                    # the previous group's softmax read ``red``; this one
                    # overwrites it with its own scores
                    gpu.barrier()
                hn = head_group * WAVES + fx.min(lane % 16, WAVES - 1)
                rgk = wave % (SPLIT_KEYS // 16)
                c = fx.Vector.filled(4, 0.0, fx.Float32)

                def qk_operands(st):
                    kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                    key = rgk * 16 + lane % 16
                    kw = (kst < KV_LORA // 32).select(
                        KT_OFF + key * KS + kst * 16,
                        PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                    )
                    a = fx.ptr_load(
                        xs + (kw + (lane // 16) * 4), result_type=v4f
                    ).bitcast(fx.BFloat16)
                    b = fx.ptr_load(
                        xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f
                    ).bitcast(fx.BFloat16)
                    return a, b

                N_QK = QK_DIM // 32 // 2
                if const_expr(_QK_PIPE):
                    qk_ab = qk_operands(0)
                    for st in range_constexpr(N_QK):
                        qk_cur = qk_ab
                        if const_expr(st + 1 < N_QK):
                            qk_ab = qk_operands(st + 1)
                        c = _mfma_bf16(qk_cur[0], qk_cur[1], c)
                else:
                    for st in range_constexpr(N_QK):
                        a, b = qk_operands(st)
                        c = _mfma_bf16(a, b, c)
                if const_expr(SPLIT_KEYS == 64) or wave < 4:
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                gpu.barrier()
                stamp("split", tt, 6)
                # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
                kidx = t * SPLIT_KEYS + lane
                valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
                r16 = lane % 16
                cl = split_score_column(wave, lane)
                key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
                half_stride = SPLIT_KEYS // 16
                raw = lds_ld(red, (key_rg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                    red, ((key_rg + half_stride) * 64 + cl) * 4 + r16 % 4
                )
                sc_v = valid.select(raw * scale, fx.Float32(NEG))
                m = wave_max(sc_v)
                p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                lsum = wave_sum(p)
                p_n = _xshfl(p, 1)
                if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                    lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                gpu.barrier()
                stamp("split", tt, 3)
                # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
                # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                for g in range_constexpr(KV_LORA // 32 // WAVES):
                    dw = (
                        wave * (KV_LORA // 32 // WAVES) + g
                    ) * 16 + lane % 16  # dim pair word
                    c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                    c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                    for js in range_constexpr(SPLIT_KEYS // 32):
                        a = fx.ptr_load(
                            pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        ws = [
                            fx.ptr_load(
                                ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)
                            ).bitcast(fx.Int32)
                            for i in range(8)
                        ]
                        w_lo = [
                            (ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16)
                            for i in range(4)
                        ]
                        w_hi = [
                            fx.Int32(fx.Uint32(ws[2 * i]) >> 16)
                            | (ws[2 * i + 1] & -65536)
                            for i in range(4)
                        ]
                        b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(
                            fx.BFloat16
                        )
                        b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(
                            fx.BFloat16
                        )
                        c0 = _mfma_bf16(a, b0, c0)
                        c1 = _mfma_bf16(a, b1, c1)
                    if lane < 32:
                        for e in range_constexpr(4):
                            hh = split_acc_head(head_group, lane // 16, e)
                            put_bf(
                                mb("sp_acc"),
                                ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2,
                                [c0[e], c1[e]],
                            )
                if lane == 0:  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
                stamp("split", tt, 4)

        def split_pre_pick(s, r):
            """(``split_rows``, pages) of row s's task in this CTA's split
            round r, from ``row_pre``."""
            pg = PRE_TABLES.index("pages") * 16 + 2 * r
            pages = (read_lane_i32(row_pre, pg), read_lane_i32(row_pre, pg + 1))
            return split_rows(s, pre=True), pages

        for tt in range(start("split"), N_SPLIT_TASKS, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s, hg0, t = split_coords(tt)
            if const_expr(_SPLIT_PRE):
                rows, pages = split_pre_pick(s, tt // G)
                if t < live_splits(rows[0]):  # CTA-uniform
                    split_task(tt, s, hg0, t, rows, pages)
            elif const_expr(_SKIP_EMPTY_SPLITS):
                # the task's own row loads, so a live split waits no longer
                rows = split_rows(s)
                if t < live_splits(rows[0]):  # CTA-uniform
                    split_task(tt, s, hg0, t, rows)
            else:
                split_task(tt, s, hg0, t)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        UV_UNITS = UV_NKC // (attention_k_chunks_per_unit * UV_WPR)

        def uv_merge_whole(s, head, n_live, tt):
            """LDS xs = the head's merged latent (bf16 pairs; PTPC: FP8 + scales)."""
            # Four 16-split quarters keep each poll and its live payload bounded.
            # Every thread owns one (quarter, half-of-latent) pair and processes
            # its two latent-pair positions sequentially; red holds quarter sums.
            SPH = N_SPLIT // 4
            dp_lo = tid % (KV_LORA // 4)
            hf = tid // (KV_LORA // 4)
            spi = fx.min(lane, n_live - 1)
            ml = (s * N_SPLIT + spi) * H + head

            def acc_split(j):
                # a skipped split reads the last live one's (cached) partial sum
                # instead, at weight 0
                if const_expr(_SKIP_EMPTY_SPLITS):
                    return s * N_SPLIT + fx.min(hf * SPH + j, n_live - 1)
                return s * N_SPLIT + hf * SPH + j

            ml_got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)])
            stamp("uv", tt, 6)
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                ok_sp = lane < n_live
                m_sp = ok_sp.select(ml_got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(ml_got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                m_all = wave_max(m_sp)
                w_sp = _exp(m_sp - m_all)
                den = wave_sum(l_sp * w_sp)
                if lane < N_SPLIT:
                    lds_st(
                        misc,
                        lane,
                        w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                    )
                if lane == 0:
                    lds_st(misc, N_SPLIT, m_all)
                    lds_st(misc, N_SPLIT + 1, den)
            gpu.barrier()
            stamp("uv", tt, 7)
            for dh in range_constexpr(2):
                dp = dp_lo + dh * (KV_LORA // 4)
                got = poll(
                    [
                        (
                            mb("sp_acc"),
                            (acc_split(j) * H + head) * (KV_LORA // 2) + dp,
                            1,
                        )
                        for j in range(SPH)
                    ]
                )
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for j in range_constexpr(SPH):
                    wj = lds_ld(misc, hf * SPH + j)
                    a0, a1 = bf2_f32(got[j][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
            gpu.barrier()
            uv_valid = tid < KV_LORA // 2
            uv0, uv1 = fx.Float32(0.0), fx.Float32(0.0)
            if uv_valid:
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(4):
                    o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                    o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                uv0, uv1 = bf16_round(o0), bf16_round(o1)
                if const_expr(not attention_ptpc):
                    lds_st(xs, tid, bf16_pair(uv0, uv1))
            if const_expr(attention_ptpc):
                uv_max = wave_max(
                    uv_valid.select(
                        fmax(fmath.absf(uv0), fmath.absf(uv1)), fx.Float32(0.0)
                    )
                )
                if (lane == 0) & (wave < KV_LORA // 128):
                    lds_st(
                        misc,
                        N_SPLIT + 2 + wave,
                        (uv_max == 0.0).select(
                            fx.Float32(1.0), uv_max * (1.0 / FP8_MAX)
                        ),
                    )
            gpu.barrier()
            if const_expr(attention_ptpc):
                if uv_valid:
                    uv_scale = lds_ld(misc, N_SPLIT + 2 + tid // 64)
                    uv_rcp = 1.0 / uv_scale
                    st_ptpc(
                        tid * 2,
                        div_rn(uv0, uv_scale, uv_rcp),
                        div_rn(uv1, uv_scale, uv_rcp),
                    )
                gpu.barrier()

        UV_QUARTERS = V_DIM // UV_TILE  # tasks a head = latent slices
        UV_QP = KV_LORA // 2 // UV_QUARTERS  # latent pairs a slice: one a lane
        UV_SPW = N_SPLIT // WAVES  # splits a wave merges

        def uv_merge_slice(s, t, head, n_live, tt):
            """LDS xs = the head's merged latent (bf16 pairs): this task merges
            latent slice q = its V tile (wave w: splits w * UV_SPW .. + UV_SPW,
            lane: a latent pair), publishes it to `lat` and gathers the others."""
            q = t % UV_QUARTERS
            k0 = wave * UV_SPW
            lat_row = (s * H + head) * (KV_LORA // 2)
            o0 = fx.Float32(0.0)
            o1 = fx.Float32(0.0)
            if k0 < n_live:  # wave-uniform: a wave of dead splits loads nothing
                # every wave weighs all the splits (lane = split) itself: no
                # barrier between the poll and the sums; a dead split reads the
                # last live one's partial sum at weight 0
                ml = (s * N_SPLIT + fx.min(lane, n_live - 1)) * H + head
                got = poll(
                    [(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)]
                    + [
                        (
                            mb("sp_acc"),
                            ((s * N_SPLIT + fx.min(k0 + j, n_live - 1)) * H + head)
                            * (KV_LORA // 2)
                            + q * UV_QP
                            + lane,
                            1,
                        )
                        for j in range(UV_SPW)
                    ]
                )
                ok_sp = lane < n_live
                m_sp = ok_sp.select(got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                m_all = wave_max(m_sp)
                w_sp = _exp(m_sp - m_all)
                den = wave_sum(l_sp * w_sp)
                w_sp = w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0))
                for j in range_constexpr(UV_SPW):
                    wj = bpermute_i32((k0 + j) * 4, w_sp.bitcast(fx.Int32)).bitcast(
                        fx.Float32
                    )
                    a0, a1 = bf2_f32(got[2 + j][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
            stamp("uv", tt, 6)
            lds_st(red, (wave * UV_QP + lane) * 2, o0)
            lds_st(red, (wave * UV_QP + lane) * 2 + 1, o1)
            gpu.barrier()
            if tid < UV_QP:
                u0 = lds_ld(red, tid * 2)
                u1 = lds_ld(red, tid * 2 + 1)
                for w in range_constexpr(1, WAVES):
                    u0 = u0 + lds_ld(red, (w * UV_QP + tid) * 2)
                    u1 = u1 + lds_ld(red, (w * UV_QP + tid) * 2 + 1)
                u0, u1 = bf16_round(u0), bf16_round(u1)
                put_bf(mb("lat"), (lat_row + q * UV_QP + tid) * 2, [u0, u1])
                lds_st(xs, q * UV_QP + tid, bf16_pair(u0, u1))
            # the other slices, two pairs a thread
            if tid < (UV_QUARTERS - 1) * (UV_QP // 2):
                p = ((q + 1 + tid // (UV_QP // 2)) % UV_QUARTERS) * UV_QP + (
                    tid % (UV_QP // 2)
                ) * 2
                v = poll([(mb("lat"), lat_row + p, 2)])[0]
                lds_st(xs, p, v[0].bitcast(fx.Float32))
                lds_st(xs, p + 1, v[1].bitcast(fx.Float32))
            gpu.barrier()
            stamp("uv", tt, 7)

        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV  # sample
            t = tt % N_UV  # global 64-row tile
            local_t = dcp_local_uv_tile(t, L)
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wuv,
                    r_suv,
                    local_t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    128,
                    attention_b_word(kc * 64),
                    scale_rows=uv_scale_rows,
                    coef=(
                        (lambda: lds_ld(misc, N_SPLIT + 2 + kc // 2))
                        if const_expr(attention_ptpc)
                        else None
                    ),
                )

            # loaded ahead of the weight prefetch: loads retire in order, so a
            # wait on it after the prefetch would wait for the whole tile
            n_live = (
                live_splits(row_nkeys(s)) if const_expr(_SKIP_EMPTY_SPLITS) else N_SPLIT
            )
            if const_expr(_SKIP_EMPTY_SPLITS):
                rocdl.sched_barrier(0)
            pre = [u_uv(c) for c in range(UV_UNITS)]
            hint_wait(
                N_SPLIT,
                lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head),
                mark=("uv", tt),
            )
            pre_poll(n_live, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            if const_expr(_UV_SLICE):
                uv_merge_slice(s, t, head, n_live, tt)
            else:
                uv_merge_whole(s, head, n_live, tt)
            # Debug stamp placed HERE, not after the per-split weights where it
            # used to sit, so ``timeline_report``'s "stage" and "compute" cut
            # this stage where the question is: everything above reconstructs
            # ONE sample's attention latent (two polls of `sp_acc`, the
            # weighted sum, the FP8 quant) and is per (sample, head); every-
            # thing below is the UV GEMM, whose weight tile depends only on
            # ``t`` and is therefore re-read identically by all S samples.
            # Merging the samples into the MFMA's spare N columns would divide
            # the second by S and leave the first alone -- but it also cuts the
            # stage to ``N_UV`` tasks, a quarter of the grid. So it pays only
            # if "compute" dominates "stage"; this stamp is what says which.
            stamp("uv", tt, 2)
            owner = dcp_merge_latent(s, t)
            if owner:
                acc = run_units(u_uv, UV_UNITS, UV_UNITS, pre)
                if const_expr(attention_ptpc):
                    reduce_rows(UV_R, acc, emit_ptpc_scalar(UV_TILE, ld_f32(r_suv, 0)))
                else:
                    reduce_rows(UV_R, acc, emit_out(UV_TILE))
                stamp("uv", tt, 3)
                gpu.barrier()
                if tid < UV_TILE // 4:
                    r = tid * 4
                    put_bf(
                        mb("o"),
                        s * O_K + local_t * UV_TILE + r,
                        [lds_ld(outs, r + j) for j in range(4)],
                    )
            stamp("uv", tt, 4)

        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        O_UNITS = O_NKC // (attention_k_chunks_per_unit * O_WPR)
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o", t, 0)

            def u_o(c):
                kc = ((wave % O_WPR) * O_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wo,
                    r_so,
                    t * O_R + wave // O_WPR,
                    kc,
                    O_NKC,
                    O_K,
                    128,
                    attention_b_word(n_sel() * O_K + kc * 64),
                )

            pre = [u_o(c) for c in range(O_UNITS)]
            hint_wait(
                S * N_UV,
                lambda k: (
                    mb("o"),
                    (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1,
                ),
                mark=("o", t),
            )
            if const_expr(attention_ptpc):
                o_scales = stage_x_pairs_ptpc("o", S, O_K, lambda k: k)
            else:
                stage_x_pairs("o", S * O_K, lambda k: k)
            stamp("o", t, 2)
            gpu.barrier()
            acc = run_units(u_o, O_UNITS, O_UNITS, pre)
            if const_expr(attention_ptpc):
                reduce_rows(O_R, acc, emit_ptpc(ROW_TILE, t * ROW_TILE, r_so, o_scales))
            else:
                reduce_rows(O_R, acc, emit_out(ROW_TILE))
            stamp("o", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            bo.buffer_load(
                                r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32
                            )
                        )
                    ],
                    fx.Int32,
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            peer_reduce(
                "attn",
                t,
                resid_h,
                lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            )
            stamp("o", t, 4)

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K.
        # S > 1 gets S times as many independent router CTAs instead of serializing
        # every sample's normalization and output columns inside one CTA.
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt if const_expr(S == 1) else tt % N_ROUTER
            router_sample = fx.Int32(0) if const_expr(S == 1) else tt // N_ROUTER
            stamp("router", tt, 0)

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.Int32(0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(
                    r_wr,
                    t * ROUTER_TILE // 16,
                    kc,
                    R_NKC,
                    (r_ns * HIDDEN + kc * 64) // 2,
                    r_ln,
                )

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (
                    mb("a"),
                    router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                ),
                mark=("router", tt),
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
            r_gp = _rsrc(g_post)
            x_blk = wave * N_ROUTER + t
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [
                    (mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks
                ]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                stamp("router", tt, 5, lead=THREADS - 64)
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=1)
            stamp("router", tt, 2)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = rstds[0]
                a0, a1 = xa[0]
                if const_expr(native_fp4_mfma):
                    q0, q1, qs = quant_mxfp8(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                else:
                    q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                w8 = (
                    fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                    & 0xFFFF
                )
                w8n = _xshfl(w8, 1)
                if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                    put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                d0, d1 = _fp8_roundtrip(q0, q1)
                qscale = (
                    (qs << 23).bitcast(fx.Float32)
                    if const_expr(native_fp4_mfma)
                    else qs
                )
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                    _rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
                if const_expr(native_fp4_mfma):
                    if lane % 16 == 0:
                        put(mb("xqs"), x_s * XQ_GROUPS + x_blk * 4 + lane // 16, qs)
                else:
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            gpu.barrier()
            stamp("router", tt, 3)
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = fx.Int32(0)
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(
                            red,
                            (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4,
                        )
                put(
                    mb("scores"), router_sample * N_EXPERTS + t * ROUTER_TILE + r, logit
                )  # ML4: softmax routing, keep the logit
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing: one whole wave per sample, in wave-sized batches."""
            for sample_batch in range_constexpr(sample_wave_batches(S)):
                route_sample = wave + sample_batch * WAVES
                if route_sample < S:
                    e, w = route_top8(route_sample, bs=bs)
                    if (
                        lane < MOE_SLOTS
                    ):  # slot 0: the shared expert, then pick lane (slot lane + 1)
                        q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(
                            keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e)
                        )
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
        UG_NKC = HIDDEN // 64
        UG_W_BYTES = 2 * expert_inter * HIDDEN // (2 if expert_mxfp4 else 1)
        UG_S_BYTES = (
            2 * expert_inter * (HIDDEN // 32)
            if expert_mxfp4
            else 2 * expert_inter // SCALE_BM * (HIDDEN // 128) * 4
        )
        hoist_shared_ptrs()

        if const_expr(S == 1):
            # Each CTA handles one eight-intermediate tile per task round;
            # the first routed slot also computes the shared expert. TP4 has
            # two rounds, TP8 one. All eight waves split the K dimension.
            UG8 = 8
            UG8_CPW = UG_NKC // WAVES
            N_UG1 = TOP_K * (
                expert_inter // UG8
            )  # TOP_K = 8 makes this == expert_inter
            if const_expr(_UG1_FUSE and N_UG1 > G):
                # TP8: 384 tasks over 256 CTAs. A CTA's tasks share one staging
                # of the activation and one routing, and the routed weight tiles
                # of all of them are in flight together instead of one round
                # after the other (round 2 used to re-stage for ~6.6 us).
                UG1_ROUNDS = (N_UG1 + G - 1) // G
                UG1_C = expert_inter // UG8
                u_first = start("ug")
                s_u = fx.Int32(0)
                tasks1 = []
                for r in range_constexpr(UG1_ROUNDS):
                    u = fx.Int32(u_first + r * G)
                    live = None if r == 0 else u < N_UG1
                    if const_expr(UG_SHIFT):
                        u = (u + UG_SHIFT) % N_UG1
                    c = u % UG1_C
                    tasks1.append(
                        dict(
                            u=u,
                            live=live,
                            c=c,
                            slot=(u < UG1_C).select(
                                fx.Int32(MOE_SLOTS - 1), u // UG1_C
                            ),
                            w_rg=((lane % 16) // 8) * (expert_inter // 16) + c // 2,
                            w_ln=(lane & -16) | ((c % 2) * 8 + lane % 8),
                            s_rg=(lane // 32) * (expert_inter // 16) + c // 2,
                        )
                    )
                # Only round 0 can carry the shared expert: later rounds start
                # at G or past it, and every logical task < UG1_C is in round 0.
                has_sh = tasks1[0]["u"] < UG1_C

                def stamp1(t, which):
                    if const_expr(timeline):
                        if const_expr(t["live"] is None):
                            stamp("ug", t["u"], which)
                        else:
                            live = t["live"]
                            if live:
                                stamp("ug", t["u"], which)

                def u_ug1(t, cc, e, live=None):
                    nw = (
                        None
                        if live is None
                        else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                    )
                    ns = (
                        None
                        if live is None
                        else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                    )
                    r_wug = bo.create_buffer_resource_from_addr(
                        expert_addr(w_ug, e, UG_W_BYTES, 0), num_records_bytes=nw
                    )
                    r_sug = bo.create_buffer_resource_from_addr(
                        expert_addr(s_ug, e, UG_S_BYTES, 1), num_records_bytes=ns
                    )
                    kc = (wave * (UG8_CPW // 2) + cc) * 2
                    wv = [
                        fx.Vector(
                            bo.buffer_load(
                                r_wug,
                                ((t["w_rg"] * UG_NKC + kc + h) * 64 + t["w_ln"]) * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for h in range(2)
                    ]
                    sc = ld_f32(
                        r_sug,
                        (t["s_rg"] * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2,
                    )
                    return (
                        "f8f8",
                        wv,
                        lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                        kc * 16 + (lane // 16) * 4,
                    )

                for t in tasks1:
                    stamp1(t, 0)
                NB = XQ_BLOCKS // WAVES
                ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]
                bs = load_bias()
                pre = [
                    u_ug1(tasks1[0], cc, fx.Int32(SHARED_EXPERT), has_sh)
                    for cc in range(UG8_CPW // 2)
                ]
                hint_wait(
                    N_ROW_TILES,
                    lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                    mark=("ug", tasks1[0]["u"]),
                )
                NQ4 = HIDDEN // (4 * THREADS)
                got = poll(
                    [
                        (mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2)
                        for i in range(NQ4)
                    ]
                    + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                )
                av = [bf2_f32(w[0]) for w in got[NQ4:]]
                ss = fx.Float32(0.0)
                for w in got[:NQ4]:
                    for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                        ss = ss + a * a
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for j in range_constexpr(NB):
                    q0, q1, qs = quant_scaled(
                        av[j][0] * rstd * gps[j][0],
                        av[j][1] * rstd * gps[j][1],
                    )
                    st_f8(ks_[j], q0, q1)
                    if lane == 0:
                        lds_st(misc, 8 + wave + j * WAVES, qs)
                # with `_DN_ROUTE_REUSE` the down stage's full table lives in
                # keys[0, MOE_SLOTS), so this CTA's own picks go past it
                KEY1 = MOE_SLOTS if const_expr(_DN_ROUTE_REUSE) else 0
                if wave == 0:
                    e, w = route_top8(s_u, bs=bs)
                    for r in range_constexpr(UG1_ROUNDS):
                        if lane == tasks1[r]["slot"] - 1:
                            lds_st(keys, KEY1 + r, e)
                            lds_st(misc, r, w)
                    if const_expr(_DN_ROUTE_REUSE):
                        if lane < MOE_SLOTS:  # `dn_route`'s table, from the same picks
                            q = (lane + 1) % MOE_SLOTS
                            lds_st(
                                keys,
                                q,
                                (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e),
                            )
                            lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))
                for t in tasks1:
                    stamp1(t, 2)
                gpu.barrier()
                e_sels = [_uniform(lds_ld(keys, KEY1 + r)) for r in range(UG1_ROUNDS)]
                posts = [
                    [u_ug1(t, cc, e_sels[r], t["live"]) for cc in range(UG8_CPW // 2)]
                    for r, t in enumerate(tasks1)
                ]
                reduce_rows(
                    1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16)
                )
                gpu.barrier()
                for r in range_constexpr(UG1_ROUNDS):

                    def ug1_task(r=r, t=tasks1[r]):
                        reduce_rows(
                            1,
                            mma_units([fx.Float32(0.0) for _ in range(4)], posts[r]),
                            lambda rl, n, v: lds_st(outs, 16 + rl, v),
                        )
                        stamp1(t, 3)
                        gpu.barrier()
                        if tid < UG8:
                            rr = (tid % (UG8 // 2)) * 2
                            o = (tid // (UG8 // 2)) * 16
                            g0, g1 = lds_ld(outs, o + rr), lds_ld(outs, o + rr + 1)
                            v0 = lds_ld(outs, o + UG8 + rr)
                            v1 = lds_ld(outs, o + UG8 + rr + 1)
                            emit = tid >= UG8 // 2
                            if const_expr(r == 0):
                                emit = has_sh | emit
                            if emit:
                                put2(
                                    mb("mid"),
                                    (tid < UG8 // 2).select(fx.Int32(0), t["slot"])
                                    * expert_inter
                                    + t["c"] * UG8
                                    + rr,
                                    g0 * _rcp(1.0 + _exp(-g0)) * v0,
                                    g1 * _rcp(1.0 + _exp(-g1)) * v1,
                                )
                        if (t["c"] == 0) & (tid == 0):  # routing record (debug / tests)
                            put(mb("sel"), t["slot"], e_sels[r])
                            put(mb("prob"), t["slot"], lds_ld(misc, r))
                            if const_expr(r == 0):
                                if has_sh:
                                    put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                                    put(mb("prob"), 0, fx.Float32(1.0))
                        stamp1(t, 4)

                    if const_expr(r == 0):
                        ug1_task()
                    else:
                        live = tasks1[r]["live"]
                        if live:  # CTA-uniform: the task's barriers stay matched
                            ug1_task()
                gpu.barrier()
            else:
                for u in range(start("ug"), N_UG1, G):
                    u = fx.Int32(u)
                    if const_expr(UG_SHIFT):
                        u = (u + UG_SHIFT) % N_UG1
                    stamp("ug", u, 0)
                    s_u, c = fx.Int32(0), u % (expert_inter // UG8)
                    has_sh = u < expert_inter // UG8
                    slot = has_sh.select(
                        fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8)
                    )
                    # the FP8 activation is computed here from the post-attention state (in
                    # parallel with the router): RMSNorm, then per-128 quant with one wave per
                    # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
                    NB = XQ_BLOCKS // WAVES
                    ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                    r_gp = _rsrc(g_post)
                    gps = [
                        (ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_
                    ]  # issued ahead of the wait
                    bs = load_bias()
                    w_rg = (
                        c
                        if paired_ug_rows
                        else ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                    )  # MFMA rows 0-7 gate, 8-15 up
                    w_ln = (
                        lane
                        if paired_ug_rows
                        else (lane & -16) | ((c % 2) * 8 + lane % 8)
                    )
                    s_rg = (lane // 32) * (
                        expert_inter // 16
                    ) + c // 2  # this lane's output rows

                    def u_ug8(
                        cc, e, live=None
                    ):  # expert e's weights (loads return 0 unless live)
                        nw = (
                            None
                            if live is None
                            else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                        )
                        ns = (
                            None
                            if live is None
                            else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                        )
                        r_wug = bo.create_buffer_resource_from_addr(
                            expert_addr(w_ug, e, UG_W_BYTES, 0), num_records_bytes=nw
                        )
                        r_sug = bo.create_buffer_resource_from_addr(
                            expert_addr(s_ug, e, UG_S_BYTES, 1), num_records_bytes=ns
                        )
                        unit = wave * (UG8_CPW // 2) + cc
                        kc = unit * 2
                        if const_expr(expert_mxfp4):
                            if const_expr(native_fp4_mfma):
                                return unit_native_mxfp4(
                                    r_wug,
                                    r_sug,
                                    w_rg,
                                    unit,
                                    HIDDEN,
                                    unit * 32,
                                    8 + unit * 4 + lane // 16,
                                    ln=w_ln,
                                )
                            return unit_mxfp4(
                                r_wug,
                                r_sug,
                                w_rg,
                                unit,
                                HIDDEN,
                                unit * 32,
                                lambda: _uniform_f32(lds_ld(misc, 8 + unit)),
                                w_ln,
                            )
                        wv = [
                            fx.Vector(
                                bo.buffer_load(
                                    r_wug,
                                    ((w_rg * UG_NKC + kc + h) * 64 + w_ln) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for h in range(2)
                        ]
                        sc = ld_f32(
                            r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2
                        )
                        return (
                            "f8f8",
                            wv,
                            lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                            kc * 16 + (lane // 16) * 4,
                        )

                    # the shared expert's weights do not depend on routing: prefetch them (the
                    # later zero-weight MMAs of the other tasks are cheaper than a branch)
                    pre = [
                        u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh)
                        for cc in range(UG8_CPW // 2)
                    ]
                    hint_wait(
                        N_ROW_TILES,
                        lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                        mark=("ug", u),
                    )
                    # the sum of squares takes the router's element partition and order
                    # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
                    NQ4 = HIDDEN // (4 * THREADS)
                    got = poll(
                        [
                            (mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2)
                            for i in range(NQ4)
                        ]
                        + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                    )
                    av = [bf2_f32(w[0]) for w in got[NQ4:]]
                    ss = fx.Float32(0.0)
                    for w in got[:NQ4]:
                        for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                            ss = ss + a * a
                    rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                    for j in range_constexpr(NB):
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(
                                av[j][0] * rstd * gps[j][0],
                                av[j][1] * rstd * gps[j][1],
                            )
                            st_mxfp8(ks_[j], q0, q1)
                            if lane % 16 == 0:
                                lds_st(
                                    misc,
                                    8 + (wave + j * WAVES) * 4 + lane // 16,
                                    qs.bitcast(fx.Float32),
                                )
                        else:
                            q0, q1, qs = quant_scaled(
                                av[j][0] * rstd * gps[j][0],
                                av[j][1] * rstd * gps[j][1],
                            )
                            st_f8(ks_[j], q0, q1)
                            if lane == 0:
                                lds_st(misc, 8 + wave + j * WAVES, qs)
                    if wave == 0:
                        e, w = route_top8(s_u, bs=bs)
                        if lane == slot - 1:
                            lds_st(keys, 0, e)
                            lds_st(misc, 0, w)
                    stamp("ug", u, 2)
                    gpu.barrier()
                    e_sel = _uniform(lds_ld(keys, 0))
                    post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], post),
                        lambda rl, n, v: lds_st(outs, 16 + rl, v),
                    )
                    stamp("ug", u, 3)
                    gpu.barrier()
                    if (
                        tid < UG8
                    ):  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                        r = (tid % (UG8 // 2)) * 2
                        o = (tid // (UG8 // 2)) * 16
                        g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                        u0, u1 = (
                            lds_ld(outs, o + UG8 + r),
                            lds_ld(outs, o + UG8 + r + 1),
                        )
                        if has_sh | (tid >= UG8 // 2):
                            put2(
                                mb("mid"),
                                (tid < UG8 // 2).select(fx.Int32(0), slot)
                                * expert_inter
                                + c * UG8
                                + r,
                                g0 * _rcp(1.0 + _exp(-g0)) * u0,
                                g1 * _rcp(1.0 + _exp(-g1)) * u1,
                            )
                    if (c == 0) & (tid == 0):  # routing record (debug / tests)
                        put(mb("sel"), slot, e_sel)
                        put(mb("prob"), slot, lds_ld(misc, 0))
                        if has_sh:
                            put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                            put(mb("prob"), 0, fx.Float32(1.0))
                    stamp("ug", u, 4)
                    if const_expr(N_UG1 > G):
                        gpu.barrier()
        elif const_expr(S > 1):
            # One eight-intermediate tile per CTA and task round. Shared-expert weights
            # feed all sample columns of one MFMA, while routed-expert weights are
            # prefetched one sample ahead.  This avoids the segmented partial
            # tiles and mailbox reduction used by the older S=2/4 schedule.
            UG8 = 8
            UG8_UNITS = (HIDDEN // 128) // WAVES
            XW = XQ_X_STRIDE
            if const_expr(not _UG_SH_EARLY):  # else `ug_flat` routes after them
                dn_route(load_bias())
                gpu.barrier()
            ug_n = fx.Int32(0)
            if const_expr(moe_dedup):
                # ---- the expert-group table.
                # Routed (sample, slot) units that name the SAME expert can share
                # one weight tile: their rows always have distinct samples (top-k
                # never picks an expert twice for one token), so they fit the
                # MFMA's own 16 N columns, which the shared expert already uses.
                # Every CTA derives the table from the ``keys`` it already holds
                # in LDS, so this costs no mailbox traffic.
                # Pass 1: per unit, is it the first of its expert, and which slot
                # does each sample contribute (one nibble, slot + 1 .. 0 = none)?
                # Threads past ``UG_ROUTED`` repeat the last unit and store
                # nothing -- the work must stay CTA-uniform to cross a barrier.
                ug_r = fx.min(tid, UG_ROUTED - 1)
                ug_e = lds_ld(keys, (ug_r // TOP_K) * MOE_SLOTS + ug_r % TOP_K + 1)
                ug_lead = fx.Int32(1)
                ug_mask = [fx.Int32(0) for _ in range(UG_MASK_WORDS)]
                for other in range_constexpr(UG_ROUTED):
                    o_sample, o_slot = other // TOP_K, other % TOP_K + 1
                    same = lds_ld(keys, o_sample * MOE_SLOTS + o_slot) == ug_e
                    ug_lead = (same & (fx.Int32(other) < ug_r)).select(
                        fx.Int32(0), ug_lead
                    )
                    word, shift = (o_sample * 4) // 32, (o_sample * 4) % 32
                    # fold to two's complement: sample 7's nibble reaches bit 28,
                    # and slot 8 there is 0x80000000, which is not a Python int32
                    nibble = (o_slot << shift) - (
                        1 << 32 if o_slot << shift >= 1 << 31 else 0
                    )
                    ug_mask[word] = ug_mask[word] | same.select(
                        fx.Int32(nibble), fx.Int32(0)
                    )
                if tid < UG_ROUTED:
                    lds_st(grp, tid, ug_lead)
                    lds_st(
                        grp, GRP_EXP + tid, fx.Int32(0)
                    )  # dead groups read a safe expert
                    for word in range_constexpr(UG_MASK_WORDS):
                        lds_st(
                            grp, GRP_MASK + tid * UG_MASK_WORDS + word, ug_mask[word]
                        )
                gpu.barrier()
                # Pass 2: compact the leads, so the dead task rounds are the tail
                # ones and a CTA can skip them whole instead of issuing an MFMA
                # chain whose weight loads return zero.
                ug_rank, ug_count = fx.Int32(0), fx.Int32(0)
                for other in range_constexpr(UG_ROUTED):
                    flag = lds_ld(grp, other)
                    ug_rank = ug_rank + (fx.Int32(other) < ug_r).select(
                        flag, fx.Int32(0)
                    )
                    ug_count = ug_count + flag
                if tid == 0:
                    lds_st(grp, GRP_N, ug_count)
                if (tid < UG_ROUTED) & (ug_lead == fx.Int32(1)):
                    lds_st(grp, GRP_EXP + ug_rank, ug_e)
                    for word in range_constexpr(UG_MASK_WORDS):
                        lds_st(
                            grp,
                            GRP_GMASK + ug_rank * UG_MASK_WORDS + word,
                            ug_mask[word],
                        )
                gpu.barrier()
                ug_n = _uniform(lds_ld(grp, GRP_N))
            stage_ug_first = atom_experts or (expert_mxfp4 and S > 4)
            if const_expr(stage_ug_first):
                stage_xq(list(range(S)))
                gpu.barrier()
            u0 = start("ug")
            UG_C = expert_inter // UG8
            # ``c`` is the same on every round of a CTA -- ``G`` is a whole number
            # of column tiles -- which is what lets the group schedule below walk
            # experts on the rounds the sample loop used to walk samples.
            assert not moe_dedup or G % UG_C == 0, (
                "the group schedule needs whole column tiles per CTA sweep"
            )
            # Same number of rounds either way: ``2 * S`` (sample) rounds become
            # ``2 * S`` (group) rounds, of which only ``ug_n`` carry work.
            # Per-sample schedule: TOP_K * UG_C tasks (top-8: == inter; ug_task_rounds assumed that).
            # They need not fill whole CTA rounds: TP8's 384 leave the last
            # round to the first UG_TASKS % G CTAs of the rotation.
            UG_TASKS = TOP_K * UG_C
            UG_ROUNDS = UG_ROUTED * UG_C // G if moe_dedup else (UG_TASKS + G - 1) // G
            ug_carry = {}

            def ug_round(task_round):
                u = fx.Int32(u0 + task_round * G)
                if const_expr(UG_SHIFT):
                    u = (u + UG_SHIFT) % UG_TASKS
                c = u % UG_C
                has_sh = u < UG_C
                group = u // UG_C  # moe_dedup: a DISTINCT expert, not a slot
                slot = (
                    fx.Int32(MOE_SLOTS - 1)
                    if moe_dedup
                    else has_sh.select(fx.Int32(MOE_SLOTS - 1), u // UG_C)
                )
                w_rg = (
                    c
                    if paired_ug_rows
                    else ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                )
                w_ln = (
                    lane if paired_ug_rows else (lane & -16) | ((c % 2) * 8 + lane % 8)
                )
                s_rg = (lane // 32) * (expert_inter // 16) + c // 2

                def ug8_units(e, sample, live=None):
                    if const_expr(live is None):
                        rw = _rsrc(expert_addr(w_ug, e, UG_W_BYTES, 0))
                        rs = _rsrc(expert_addr(s_ug, e, UG_S_BYTES, 1))
                    else:
                        rw = bo.create_buffer_resource_from_addr(
                            expert_addr(w_ug, e, UG_W_BYTES, 0),
                            num_records_bytes=live.select(
                                fx.Int32(UG_W_BYTES), fx.Int32(0)
                            ),
                        )
                        rs = bo.create_buffer_resource_from_addr(
                            expert_addr(s_ug, e, UG_S_BYTES, 1),
                            num_records_bytes=live.select(
                                fx.Int32(UG_S_BYTES), fx.Int32(0)
                            ),
                        )
                    sn = n_sel() if sample is None else fx.Int32(sample)
                    units = []
                    pair_word, pair_byte = None, None
                    for cc in range_constexpr(UG8_UNITS):
                        unit = wave * UG8_UNITS + cc
                        kc = unit * 2
                        # One dword holds the MXFP4 weight scales of two
                        # consecutive k-chunks (``atom_mxfp4_scale_pair``), so
                        # load it once per even unit instead of gathering one
                        # byte per unit.  ``unit`` and ``cc`` have the same
                        # parity because UG8_UNITS is even, which is what makes
                        # the pairing a compile-time decision.  Bit-identical:
                        # the same bytes, read in one instruction instead of
                        # two.  This matters here and not in ``down`` because
                        # ``ug``'s scale plane is K=HIDDEN wide, so each unit's
                        # 64 lanes touch eight cache lines for 64 bytes.
                        ug_scale = None
                        if const_expr(_UG_SCALE_PAIR and UG8_UNITS % 2 == 0):
                            if const_expr(cc % 2 == 0):
                                word, pair_byte = atom_mxfp4_scale_pair(
                                    w_rg * 16 + w_ln % 16,
                                    unit * 4 + w_ln // 16,
                                    HIDDEN // 32,
                                )
                                pair_word = fx.Int32(
                                    bo.buffer_load(rs, word, vec_width=1, dtype=T.i32)
                                )
                            ug_scale = pair_word.shrui(  # type: ignore[union-attr]
                                (pair_byte + (cc % 2) * 2) * 8
                            ) & fx.Int32(0xFF)
                        if const_expr(expert_mxfp4):
                            if const_expr(native_fp4_mfma):
                                units.append(
                                    unit_native_mxfp4(
                                        rw,
                                        rs,
                                        w_rg,
                                        unit,
                                        HIDDEN,
                                        sn * XW + unit * 32,
                                        8 + sn * XQ_S_STRIDE + unit * 4 + lane // 16,
                                        ln=w_ln,
                                        scale=ug_scale,
                                    )
                                )
                            else:
                                units.append(
                                    unit_mxfp4(
                                        rw,
                                        rs,
                                        w_rg,
                                        unit,
                                        HIDDEN,
                                        sn * XW + unit * 32,
                                        lambda unit=unit, sn=sn: lds_ld(
                                            misc, 8 + sn * XQ_S_STRIDE + unit
                                        ),
                                        w_ln,
                                    )
                                )
                            continue
                        wv = [
                            fx.Vector(
                                bo.buffer_load(
                                    rw,
                                    ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for j in range(2)
                        ]
                        sc = ld_f32(
                            rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2
                        )
                        units.append(
                            (
                                "f8f8",
                                wv,
                                lambda sc=sc, kc=kc, sn=sn: (
                                    sc * lds_ld(misc, 8 + sn * XQ_S_STRIDE + kc // 2)
                                ),
                                sn * XW + kc * 16 + (lane // 16) * 4,
                            )
                        )
                    return units

                def ug8_emit(sample, shared):
                    if tid < (S if shared else 1) * UG8 // 2:
                        n = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        g0, g1 = (
                            lds_ld(outs, n * 16 + r),
                            lds_ld(outs, n * 16 + r + 1),
                        )
                        v0 = lds_ld(outs, n * 16 + UG8 + r)
                        v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                        sn = n if shared else fx.Int32(sample)
                        sl = fx.Int32(0) if shared else slot
                        put2(
                            mb("mid"),
                            (sn * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                    if (c == 0) & (tid < S if shared else tid == 0):
                        sn = tid if shared else fx.Int32(sample)
                        sl = fx.Int32(0) if shared else slot
                        put(
                            mb("sel"),
                            sn * MOE_SLOTS + sl,
                            lds_ld(keys, sn * MOE_SLOTS + sl),
                        )
                        put(
                            mb("prob"),
                            sn * MOE_SLOTS + sl,
                            lds_ld(dnw, sn * MOE_SLOTS + sl),
                        )

                def group_slot(gi, sample):
                    """Slot `sample` routes to in group `gi`, or 0 for none."""
                    word = lds_ld(grp, GRP_GMASK + gi * UG_MASK_WORDS + sample // 8)
                    return word.shrui((sample % 8) * 4) & fx.Int32(0xF)

                def ug8_emit_group(gi):
                    """Publish one MFMA's S columns to their own (sample, slot)."""
                    if tid < S * UG8 // 2:
                        n = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                        v0 = lds_ld(outs, n * 16 + UG8 + r)
                        v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                        sl = group_slot(gi, n)
                        if sl > fx.Int32(0):
                            put2(
                                mb("mid"),
                                (n * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                                g0 * _rcp(1.0 + _exp(-g0)) * v0,
                                g1 * _rcp(1.0 + _exp(-g1)) * v1,
                            )
                    if (c == 0) & (tid < S):
                        sl = group_slot(gi, tid)
                        if sl > fx.Int32(0):
                            put(
                                mb("sel"),
                                tid * MOE_SLOTS + sl,
                                lds_ld(keys, tid * MOE_SLOTS + sl),
                            )
                            put(
                                mb("prob"),
                                tid * MOE_SLOTS + sl,
                                lds_ld(dnw, tid * MOE_SLOTS + sl),
                            )

                if const_expr(not moe_dedup or task_round == 0):
                    # ``has_sh`` can only be true on the first round either way:
                    # ``u0 < G`` and every later round adds ``G >= UG_C``.
                    shared_pre = ug8_units(fx.Int32(SHARED_EXPERT), None, has_sh)
                if const_expr(moe_dedup):
                    # Issue the NEXT round's weight tile before consuming this
                    # one, which is what the per-row schedule's sample-ahead
                    # prefetch did.  Without it each round pays the full weight
                    # load latency and the stage gets SLOWER, dedup or not --
                    # measured at +3.6% / +1.5% before this was restored.
                    live = group < ug_n
                    if const_expr(task_round == 0):
                        cur = ug8_units(
                            _uniform(lds_ld(grp, GRP_EXP + group)), None, live
                        )
                    else:
                        cur = ug_carry.pop("ahead")  # issued by the previous round
                    if const_expr(task_round + 1 < UG_ROUNDS):
                        ahead = group + G // UG_C
                        ug_carry["ahead"] = ug8_units(
                            _uniform(lds_ld(grp, GRP_EXP + ahead)), None, ahead < ug_n
                        )
                else:
                    # ``ug_prefetch`` samples of weight tiles in flight. The
                    # per-sample schedule has always kept exactly one, so each
                    # iteration's loads are issued one iteration ahead of their
                    # use -- which looked like enough, since an iteration is
                    # ~1.8 us. The timeline says otherwise: with the barrier at
                    # 0.0 us and the mailbox write at 0.2, the cost is 0.6 us
                    # issuing loads plus 1.0 us in the MFMA chain where their
                    # wait lands. And `ug` is the one stage that scales
                    # SUPER-linearly in S (2.25x against down's 1.75x), with
                    # its per-iteration cost rising 1.6 -> 1.8 us as the
                    # distinct-expert working set grows 38 -> 61 and the loads
                    # start missing L2. `down` already answers that with depth
                    # (`down_prefetch_batch`); this is the same lever for `ug`.
                    #
                    # Bit-identical, for the same reason `down_prefetch_batch`
                    # is: only the ISSUE order moves. ``mma_units`` consumes
                    # each sample's units in index order whatever the depth, so
                    # no load, MFMA or accumulation changes place.
                    ug_q = [
                        ug8_units(_uniform(lds_ld(keys, i * MOE_SLOTS + slot)), i)
                        for i in range(min(ug_prefetch, S))
                    ]
                if const_expr(not stage_ug_first and task_round == 0):
                    # Native weights can be prefetched while router activations
                    # arrive. ATOM tiles retain their lower-register staging order.
                    stage_xq(list(range(S)))
                    gpu.barrier()
                if const_expr(not moe_dedup or task_round == 0):
                    if has_sh:
                        reduce_rows(
                            1,
                            mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                            emit_out(16),
                        )
                        gpu.barrier()
                        ug8_emit(0, True)
                if const_expr(moe_dedup):
                    # One MFMA per DISTINCT expert: its weight tile is read once
                    # and every sample routed to it takes one of the 16 N columns.
                    if live:
                        stamp("ug", u, 0)
                        reduce_rows(
                            1,
                            mma_units([fx.Float32(0.0) for _ in range(4)], cur),
                            emit_out(16),
                        )
                        gpu.barrier()
                        ug8_emit_group(group)
                        stamp("ug", u, 4)
                else:
                    for sample in range_constexpr(S):
                        # Stamps 1-3 were unused, so `timeline_report` showed this
                        # stage as `hint 0 / stage 0 / compute 0 / epi 1.6` and the
                        # 1.6 us was read as an epilogue. It is not: with only 0
                        # and 4 written, the inherited columns collapse and the
                        # last bucket carries the WHOLE per-sample iteration --
                        # the stage's entire 2 * S-per-CTA cost, mislabelled. Split
                        # it where the four candidate causes separate: issuing the
                        # next sample's weight loads, the MFMA chain and its LDS
                        # reduction, the CTA barrier, and the `mid` mailbox write.
                        # One timeline row a task: column 0 is the first
                        # sample's start and 4 the last one's end, 1-3 the last
                        # sample's split.
                        if const_expr(sample == 0):
                            stamp("ug", u, 0)
                        pre = ug_q.pop(0)
                        ahead = sample + ug_prefetch
                        if const_expr(ahead < S):
                            ug_q.append(
                                ug8_units(
                                    _uniform(lds_ld(keys, ahead * MOE_SLOTS + slot)),
                                    ahead,
                                )
                            )
                        stamp("ug", u, 1)  # weight loads issued
                        # Cut BETWEEN the MFMA chain and `reduce_rows`, which
                        # carries a barrier of its own. Folded together they
                        # read as one 1.0 us bucket that was taken for load
                        # latency -- and prefetch depth, measured, does nothing
                        # (1/2/3 all inside the 0.14% floor at both widths)
                        # even though the disassembly shows it granting 54%
                        # more outstanding loads. So `ug` is NOT latency-bound,
                        # and the remaining suspect is that barrier: eight
                        # waves holding eight different experts do not arrive
                        # together, and the barrier costs the slowest. Skew is
                        # a load-balancing problem, not a prefetch one.
                        ug_acc = mma_units([fx.Float32(0.0) for _ in range(4)], pre)
                        stamp("ug", u, 2)  # MFMA chain alone
                        reduce_rows(1, ug_acc, emit_out(16))
                        gpu.barrier()
                        stamp("ug", u, 3)  # reduction + both barriers
                        ug8_emit(sample, False)
                        if const_expr(sample == S - 1):
                            stamp("ug", u, 4)  # mid mailbox write
                if const_expr(task_round + 1 < UG_ROUNDS):
                    gpu.barrier()

            def ug_flat():
                """One stream of (sample, slot, column tile) units per CTA.

                The task rounds give a CTA S tiles per round, so TP8's
                384 tasks leave the second round on half the CTAs -- and a
                CTA streams at about the same rate whether its neighbours are
                busy or not, so that round runs at half the bandwidth. Units
                0 .. UG_C - 1 are the shared expert's column tiles (all S
                samples in one MFMA), the rest the routed (sample, slot, c)
                tiles; CTA b takes units b, b + G, ... with the next unit's
                tile in flight while it computes the current one. Every unit
                is the same MFMA chain, reduction and emit as in `ug_round`.
                """
                N_FLAT = UG_C + S * UG_TASKS
                R_FLAT = (N_FLAT + G - 1) // G
                # _UG_SH_EARLY: the routed units alone, rotated so that a CTA's
                # first one is the unit it ran after its shared one before
                N_RT = S * UG_TASKS
                R_RT = (N_RT + G - 1) // G
                h0 = (u0 + (G - UG_C)) & (G - 1)

                def ug_groups_wave():
                    """The step's expert groups: group g is the g-th distinct
                    routed expert in id order, stored at `grp[g]`.

                    Lane p < UG_ROUTED takes routed pick p (sample p // TOP_K,
                    slot p % TOP_K + 1); its expert's group is the popcount of
                    the wave's expert bitmap below it. Every wave stores the
                    same words, so none waits on another. Returns the count."""
                    p = fx.min(lane, UG_ROUTED - 1)
                    e = lds_ld(keys, (p // TOP_K) * MOE_SLOTS + p % TOP_K + 1)
                    ew, eb = e.shrui(fx.Int32(5)), e & 31
                    words = [
                        wave_or_dpp((ew == w).select(fx.Int32(1) << eb, fx.Int32(0)))
                        for w in range(N_EXPERTS // 32)
                    ]
                    counts = [popcount(word) for word in words]
                    own, before = words[0], fx.Int32(0)
                    for w in range_constexpr(1, N_EXPERTS // 32):
                        own = (ew == w).select(words[w], own)
                        before = before + (ew >= w).select(counts[w - 1], fx.Int32(0))
                    gi = before + popcount(own & ((fx.Int32(1) << eb) - 1))
                    if lane < UG_ROUTED:
                        lds_st(grp, gi, e)
                    return sum(counts[1:], counts[0])

                def flat_task(i, kind="mixed"):
                    """Unit i of this CTA. ``kind``: "mixed" walks the flat list,
                    "shared" is the CTA's shared unit (``u0 < UG_C``), "routed"
                    unit i of the routed-only list and "group" unit i of the
                    (expert group, column tile) list (``_UG_DEDUP``)."""
                    if const_expr(kind == "group"):
                        h = fx.Int32(h0 + i * G)
                        live = h < grp_n * UG_C
                        sh_ok, is_sh = False, fx.Boolean(False)
                        smp = fx.min(h // UG_C, UG_ROUTED - 1)  # the group
                        c, slot = h % UG_C, None
                        e = _uniform(lds_ld(grp, smp))
                        sn = n_sel()
                    else:
                        if const_expr(kind == "routed"):
                            h = fx.Int32(h0 + i * G)
                            live = None if (i + 1) * G <= N_RT else h < N_RT
                            sh_ok, is_sh = False, fx.Boolean(False)
                        else:
                            g = fx.Int32(u0 + i * G)
                            live = None if (i + 1) * G <= N_FLAT else g < N_FLAT
                            # only a CTA's first unit can be a shared one: g >= G > UG_C after it
                            sh_ok = i == 0 and UG_C <= G
                            is_sh = (
                                (g < UG_C) if const_expr(sh_ok) else fx.Boolean(False)
                            )
                            h = (g >= UG_C).select(g - UG_C, fx.Int32(0))
                        smp = fx.min(h // UG_TASKS, S - 1)
                        rr = h % UG_TASKS
                        c = rr % UG_C
                        slot = rr // UG_C + 1
                        if const_expr(sh_ok):
                            c = is_sh.select(g, c)
                            slot = is_sh.select(fx.Int32(0), slot)
                        if const_expr(kind == "shared"):  # ahead of `dn_route`'s table
                            e = fx.Int32(SHARED_EXPERT)
                        else:
                            # keys slot 0 is the shared expert, as `dn_route` stores it
                            e = _uniform(lds_ld(keys, smp * MOE_SLOTS + slot))
                        sn = is_sh.select(n_sel(), smp) if const_expr(sh_ok) else smp
                    w_rg = ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                    w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                    s_rg = (lane // 32) * (expert_inter // 16) + c // 2
                    ro = _ROUTED_ADDR and kind in ("routed", "group")
                    if const_expr(live is None):
                        rw = _rsrc(expert_addr(w_ug, e, UG_W_BYTES, 0, ro))
                        rs = _rsrc(expert_addr(s_ug, e, UG_S_BYTES, 1, ro))
                    else:
                        rw = bo.create_buffer_resource_from_addr(
                            expert_addr(w_ug, e, UG_W_BYTES, 0, ro),
                            num_records_bytes=live.select(
                                fx.Int32(UG_W_BYTES), fx.Int32(0)
                            ),
                        )
                        rs = bo.create_buffer_resource_from_addr(
                            expert_addr(s_ug, e, UG_S_BYTES, 1, ro),
                            num_records_bytes=live.select(
                                fx.Int32(UG_S_BYTES), fx.Int32(0)
                            ),
                        )
                    # scales before weights, pinned: loads retire in order, so a
                    # wait on a scale issued after the tile would drain the tile
                    scs = [
                        ld_f32(
                            rs,
                            (s_rg * 16 // SCALE_BM) * (HIDDEN // 128)
                            + wave * UG8_UNITS
                            + cc,
                        )
                        for cc in range(UG8_UNITS)
                    ]
                    rocdl.sched_barrier(0)
                    units = []
                    for cc in range_constexpr(UG8_UNITS):
                        kc = (wave * UG8_UNITS + cc) * 2
                        wv = [
                            fx.Vector(
                                bo.buffer_load(
                                    rw,
                                    ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for j in range(2)
                        ]
                        units.append(
                            (
                                "f8f8",
                                wv,
                                lambda sc=scs[cc], kc=kc, sn=sn: (
                                    sc * lds_ld(misc, 8 + sn * XQ_S_STRIDE + kc // 2)
                                ),
                                sn * XW + kc * 16 + (lane // 16) * 4,
                            )
                        )
                    return dict(
                        units=units,
                        live=live,
                        sh_ok=sh_ok,
                        is_sh=is_sh,
                        smp=smp,
                        slot=slot,
                        c=c,
                        e=e,
                        kind=kind,
                    )

                def group_emit(t):
                    """`flat_emit` of a group unit: column n is sample n, kept
                    only if the sample routes to the group's expert, in the
                    slot it routes it from."""
                    c, e = t["c"], t["e"]

                    def slot_of(n):
                        """Sample n's `keys` slot of the group's expert, 0 if none
                        (top-k names an expert at most once a sample)."""
                        got = fx.Int32(0)
                        for k in range_constexpr(1, TOP_K + 1):
                            hit = lds_ld(keys, n * MOE_SLOTS + k) == e
                            got = hit.select(fx.Int32(k), got)
                        return got

                    n = fx.min(tid // (UG8 // 2), S - 1)
                    r = (tid % (UG8 // 2)) * 2
                    rn = fx.min(tid, S - 1)
                    gs, rs_ = slot_of(n), slot_of(rn)
                    if (tid < S * UG8 // 2) & (gs != 0):
                        g0, g1 = (
                            lds_ld(outs, n * 16 + r),
                            lds_ld(outs, n * 16 + r + 1),
                        )
                        v0 = lds_ld(outs, n * 16 + UG8 + r)
                        v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                        put2(
                            mb("mid"),
                            (n * MOE_SLOTS + gs) * expert_inter + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                    if (c == 0) & (tid < S) & (rs_ != 0):
                        put(
                            mb("sel"),
                            rn * MOE_SLOTS + rs_,
                            lds_ld(keys, rn * MOE_SLOTS + rs_),
                        )
                        put(
                            mb("prob"),
                            rn * MOE_SLOTS + rs_,
                            lds_ld(dnw, rn * MOE_SLOTS + rs_),
                        )

                def flat_emit(t):
                    if const_expr(t["kind"] == "group"):
                        return group_emit(t)
                    sh_ok, is_sh = t["sh_ok"], t["is_sh"]
                    smp, slot, c = t["smp"], t["slot"], t["c"]
                    if const_expr(sh_ok):
                        n_act = is_sh.select(fx.Int32(S * UG8 // 2), fx.Int32(UG8 // 2))
                        n_rec = is_sh.select(fx.Int32(S), fx.Int32(1))
                    else:
                        n_act, n_rec = fx.Int32(UG8 // 2), fx.Int32(1)
                    if tid < n_act:
                        n = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        g0, g1 = (
                            lds_ld(outs, n * 16 + r),
                            lds_ld(outs, n * 16 + r + 1),
                        )
                        v0 = lds_ld(outs, n * 16 + UG8 + r)
                        v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                        sn = is_sh.select(n, smp) if const_expr(sh_ok) else smp
                        put2(
                            mb("mid"),
                            (sn * MOE_SLOTS + slot) * expert_inter + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                    if (c == 0) & (tid < n_rec):
                        sn = is_sh.select(tid, smp) if const_expr(sh_ok) else smp
                        if const_expr(
                            t["kind"] == "shared"
                        ):  # what `dn_route` will store
                            rec_sel = fx.Int32(SHARED_EXPERT)
                            rec_prob = fx.Float32(1.0)
                        else:
                            rec_sel = lds_ld(keys, sn * MOE_SLOTS + slot)
                            rec_prob = lds_ld(dnw, sn * MOE_SLOTS + slot)
                        put(mb("sel"), sn * MOE_SLOTS + slot, rec_sel)
                        put(mb("prob"), sn * MOE_SLOTS + slot, rec_prob)

                def flat_body(t, i):
                    if const_expr(i == 0):
                        stamp("ug", u0, 0)
                    stamp("ug", u0, 1)
                    acc = mma_units([fx.Float32(0.0) for _ in range(4)], t["units"])
                    stamp("ug", u0, 2)
                    reduce_rows(1, acc, emit_out(16))
                    gpu.barrier()
                    stamp("ug", u0, 3)
                    flat_emit(t)
                    stamp("ug", u0, 4)

                if const_expr(_UG_SH_EARLY):
                    ug_bs = load_bias()
                    if u0 < UG_C:  # CTA-uniform: the unit's barriers stay matched
                        t_sh = flat_task(0, "shared")
                        stage_xq(list(range(S)))
                        gpu.barrier()
                        flat_body(t_sh, -1)
                    dn_route(ug_bs)
                    gpu.barrier()
                    kind, n_rounds = "routed", R_RT
                    if const_expr(_UG_DEDUP):
                        grp_n = ug_groups_wave()
                        kind = "group"
                    flat_q = [flat_task(0, kind)]
                    if u0 >= UG_C:  # the shared CTAs staged it above
                        stage_xq(list(range(S)))
                else:
                    kind, n_rounds = "mixed", R_FLAT
                    flat_q = [flat_task(0, kind)]
                    stage_xq(list(range(S)))
                gpu.barrier()

                def group_round(i, t):
                    """Round i and, nested in its liveness, the rounds after it.

                    How many units a CTA has depends on the step's group count,
                    so most CTAs end rounds early. A dead round issues its
                    prefetch all the same (out of range, zeros), and `down`'s
                    first barrier waits on those loads: 2.8 us at S=8 with
                    rounds in a flat loop. Nesting stops a CTA one prefetch
                    after its last unit."""
                    if const_expr(i + 1 < n_rounds):
                        t_next = flat_task(i + 1, kind)
                    flat_body(t, i)
                    if const_expr(i + 1 < n_rounds):
                        if t_next["live"]:  # CTA-uniform
                            group_round(i + 1, t_next)

                if const_expr(kind == "group"):
                    t_first = flat_q.pop(0)
                    if t_first["live"]:  # CTA-uniform
                        group_round(0, t_first)
                for i in range_constexpr(0 if kind == "group" else n_rounds):
                    if const_expr(i + 1 < n_rounds):
                        flat_q.append(flat_task(i + 1, kind))
                    t = flat_q.pop(0)
                    live = t["live"]
                    if const_expr(live is None):
                        flat_body(t, i)
                    else:
                        if live:  # CTA-uniform: the unit's barriers stay matched
                            flat_body(t, i)

            if const_expr(_UG_FLAT):
                ug_flat()
            else:
                for task_round in range_constexpr(UG_ROUNDS):
                    if const_expr(moe_dedup or (task_round + 1) * G <= UG_TASKS):
                        ug_round(task_round)
                    else:
                        # CTA-uniform: the round's barriers stay matched
                        if fx.Int32(u0 + task_round * G) < UG_TASKS:
                            ug_round(task_round)

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        # 2 row groups x (sample tile * 9 slots * 4) chunks: 4 waves per group.
        # S=8 is evaluated as two four-sample groups so its FP8 mid tile and
        # in-flight weight batches fit comfortably in LDS/VGPRs.
        DN_NKC = expert_inter // 64
        DN_R = (
            (DN_TILE + 15) // 16
        )  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_WPR = WAVES // DN_R
        DN_W_BYTES = HIDDEN * expert_inter // (2 if expert_mxfp4 else 1)
        DN_S_BYTES = (
            HIDDEN * (expert_inter // 32)
            if expert_mxfp4
            else HIDDEN // SCALE_BM * (expert_inter // 128) * 4
        )
        # ---- the hoisted shared expert (``dn_shared_hoist``)
        # Slot 0 names the shared expert for *every* sample -- ``dn_route``
        # stores ``SHARED_EXPERT`` at ``keys[s * MOE_SLOTS]`` and weight 1.0 at
        # ``dnw[s * MOE_SLOTS]`` for all s -- so the S units that carry it read
        # one and the same weight tile and each discard 15 of the MFMA's 16 N
        # columns. ``ug`` already fills those columns for this exact expert
        # (``ug8_units(SHARED_EXPERT, None, ...)``, whose ``sn = n_sel()`` is
        # the primitive copied here); ``down`` never did. Give a wave one unit
        # per k-chunk pair instead of one per (sample, k-chunk pair). At S=12
        # that is 48 of the 432 units replaced by 16: a 7.4% cut, and the
        # expert id becomes a build-time constant, so the LDS read, the
        # readfirstlane and both 64-bit multiplies ``DN_HOIST`` works to
        # amortise are gone outright for this slot.
        #
        # **Bit-identical, and that is what pins the shape.** Output column n
        # lives in the lanes with ``lane % 16 == n`` and accumulates the units
        # in issue order; a unit belonging to another sample contributes an
        # exact 0.0 there and is a no-op. Column n's non-zero run is therefore
        # ``slot 0, slot 1 .. slot TOP_K`` of sample n, inside the single wave
        # that owns that sample. Two conditions keep that run intact:
        #   (a) the shared units are issued FIRST, so slot 0 still leads;
        #   (b) each wave covers exactly the samples it covered before, so no
        #       term moves into another wave's partial and changes where
        #       ``reduce_rows`` sums it.
        # (b) holds only when the per-wave split lands on a sample boundary,
        # i.e. ``S % DN_WPR == 0``: true at S=12 (3 samples a wave), false at
        # S=6 (1.5), which therefore keeps the old schedule. Widen this only
        # with a measurement that re-checks bit-exactness, not by relaxing the
        # test.
        DN_SHARED = bool(dn_shared_hoist) and S > 1 and S % DN_WPR == 0
        DN_SPW = S // DN_WPR if DN_SHARED else 0  # samples a wave owns
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            if const_expr(
                ug_split(S, expert_inter) is None and not _DN_ROUTE_REUSE
            ):  # retain the tuned down-stage route schedule
                dn_route(load_bias())
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
            # (same cache lines) and are dropped in the output
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            # The indexed stream: every (sample, slot) normally, the TOP_K
            # routed slots only once the shared expert is hoisted out of it.
            # ``DN_BLK`` and ``misc`` keep the full (s, MOE_SLOTS, k) mid
            # layout either way -- the hoist moves units, not activations.
            DN_SLOTS = TOP_K if DN_SHARED else MOE_SLOTS
            DN_SLOT0 = 1 if DN_SHARED else 0  # stream slot 0 -> `keys` slot
            DN_NU = S * DN_SLOTS * DN_NKC // 2
            DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
            DN_BLK = S * MOE_SLOTS * expert_inter // 128

            # A (sample, slot) pair owns DN_NKC // 2 consecutive units -- the
            # k-chunks of one expert -- and every one of them resolves the SAME
            # expert: the same `keys` LDS read, the same readfirstlane that
            # makes the id scalar, the same two 64-bit multiplies and the same
            # two buffer-resource builds. Do it once per run of DN_HOIST units
            # instead of once per unit. Bit-identical: no load, no MFMA and no
            # accumulation moves; only redundant address arithmetic is dropped.
            #
            # DN_HOIST is a gcd because a wave's units start at
            # (wave % DN_WPR) * DN_UPW, so a run of DN_HOIST units is inside
            # one (sample, slot) only when DN_HOIST divides DN_UPW as well as
            # DN_NKC // 2. On a TP4 top-8 shard: 4 at S=12 (DN_UPW 108), 2 at S=6 (54).
            # The partial-tail case (`live` masking, DN_NU % DN_WPR != 0) opts
            # out, because there `num_records_bytes` is per-unit.
            DN_HOIST = (
                math.gcd(DN_NKC // 2, DN_UPW)
                if (_DN_HOIST and DN_NU % DN_WPR == 0)
                else 1
            )
            dn_experts: dict[int, tuple] = {}

            def dn_expert(cc):
                """`(s_q, slot_q, wb, sb)` for the (sample, slot) owning `cc`."""
                group = cc // DN_HOIST
                got = dn_experts.get(group)
                if got is not None:
                    return got
                qu = (wave % DN_WPR) * DN_UPW + group * DN_HOIST
                live = qu < DN_NU
                q = fx.min(qu, DN_NU - 1) * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (DN_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % DN_SLOTS + DN_SLOT0
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    expert_addr(w_dn, e, DN_W_BYTES, 2, _ROUTED_ADDR and DN_SHARED),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    expert_addr(s_dn, e, DN_S_BYTES, 3, _ROUTED_ADDR and DN_SHARED),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                    ),
                )
                got = (s_q, slot_q, wb, sb)
                dn_experts[group] = got
                return got

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                unit = fx.min(qu, DN_NU - 1)
                q = unit * 2  # 64-k chunk index over (s, slot, kc) of the stream
                kc = q % DN_NKC
                s_q, slot_q, wb, sb = dn_expert(cc)
                # The mid activation and ``misc`` are addressed in the full
                # (s, MOE_SLOTS, kc) space, which the stream only coincides
                # with while it still carries slot 0.
                if const_expr(DN_SHARED):
                    unit = (s_q * MOE_SLOTS + slot_q) * (DN_NKC // 2) + kc // 2
                    q = unit * 2

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(
                        _uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0)
                    )

                if const_expr(expert_mxfp4):

                    def bf16_coef():
                        return (lane % 16 == s_q).select(
                            _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                            fx.Float32(0.0),
                        )

                    if const_expr(native_fp4_mfma):
                        return unit_native_mxfp4(
                            wb,
                            sb,
                            dn_rg + gu,
                            unit % (expert_inter // 128),
                            expert_inter,
                            unit * 32,
                            unit * 4 + lane // 16,
                            bf16_coef,
                            dn_ln,
                        )
                    return unit_mxfp4_bf16(
                        wb,
                        sb,
                        dn_rg + gu,
                        unit % (expert_inter // 128),
                        expert_inter,
                        unit * 64,
                        bf16_coef,
                        dn_ln,
                    )
                return unit_f8f8(
                    wb, sb, dn_rg + gu, kc, DN_NKC, expert_inter, q * 16, coef, dn_ln
                )

            def u_dn_shared(cc):
                """One k-chunk pair of the shared expert, covering every sample
                column this wave owns in a single MFMA.

                The mirror of `u_dn`: same weight tile row group, same mid
                activation layout, but the sample comes from the lane's own
                output column (`n_sel()`, exactly as `ug8_units` does for
                this expert) instead of from the unit index, and the expert id
                is the constant `SHARED_EXPERT` rather than an LDS lookup.
                Columns outside this wave's sample range take coefficient 0.0,
                which keeps each wave's partial sum over exactly the samples it
                summed before -- see `DN_SHARED`'s bit-exactness argument.
                """
                kc = cc * 2
                sn = n_sel()  # this lane's sample column, clamped past S
                base = (wave % DN_WPR) * DN_SPW
                # `lane % 16`, not `sn`: columns at or past S must stay out of
                # the range test, and n_sel() clamps them onto sample S - 1.
                mine = (lane % 16 >= base) & (lane % 16 < base + DN_SPW)
                unit = sn * MOE_SLOTS * (DN_NKC // 2) + cc  # slot 0, mid space
                q = unit * 2

                def coef():
                    return mine.select(lds_ld(misc, unit), fx.Float32(0.0))

                if const_expr(expert_mxfp4):

                    def bf16_coef():
                        return mine.select(lds_ld(dnw, sn * MOE_SLOTS), fx.Float32(0.0))

                    if const_expr(native_fp4_mfma):
                        return unit_native_mxfp4(
                            dn_sh_wb,  # type: ignore[has-type]
                            dn_sh_sb,  # type: ignore[has-type]
                            dn_rg + gu,
                            cc,
                            expert_inter,
                            unit * 32,
                            unit * 4 + lane // 16,
                            bf16_coef,
                            dn_ln,
                        )
                    return unit_mxfp4_bf16(
                        dn_sh_wb,  # type: ignore[has-type]
                        dn_sh_sb,  # type: ignore[has-type]
                        dn_rg + gu,
                        cc,
                        expert_inter,
                        unit * 64,
                        bf16_coef,
                        dn_ln,
                    )
                return unit_f8f8(
                    dn_sh_wb,  # type: ignore[has-type]
                    dn_sh_sb,  # type: ignore[has-type]
                    dn_rg + gu,
                    kc,
                    DN_NKC,
                    expert_inter,
                    q * 16,
                    coef,
                    dn_ln,
                )

            if const_expr(DN_SHARED):
                # a build-time address: no `keys` read, no readfirstlane, and
                # one resource build per tile instead of one per DN_HOIST run
                if const_expr(separate_shared):
                    dn_sh_wb = _rsrc(shared_ptr(2))
                    dn_sh_sb = _rsrc(shared_ptr(3))
                else:
                    dn_sh_wb = _rsrc(
                        w_dn + fx.Int64(SHARED_EXPERT) * fx.Int64(DN_W_BYTES)
                    )
                    dn_sh_sb = _rsrc(
                        s_dn + fx.Int64(SHARED_EXPERT) * fx.Int64(DN_S_BYTES)
                    )
            if const_expr(_DN_MID8):
                # this tile's share of the quantization: one block a wave
                for j in range_constexpr((DN_BLK + N_DN_TILES - 1) // N_DN_TILES):
                    q_blk = t + j * N_DN_TILES
                    if (wave == j) & (q_blk < DN_BLK):  # wave-uniform
                        m8_v = get2(mb("mid"), q_blk * 128 + lane * 2)
                        m8_q0, m8_q1, m8_qs = quant_scaled(m8_v[0], m8_v[1])
                        m8_w = (
                            fx.Int32(
                                rocdl.cvt_pk_fp8_f32(
                                    T.i32, m8_q0, m8_q1, fx.Int32(0), False
                                )
                            )
                            & 0xFFFF
                        )
                        m8_wn = _xshfl(m8_w, 1)
                        if lane % 2 == 0:  # bytes 2 lane .. 2 lane + 3
                            put(
                                mb("mid8"), q_blk * 32 + lane // 2, m8_w | (m8_wn << 16)
                            )
                        if lane == 0:
                            put(
                                mb("mid8s"),
                                q_blk,
                                m8_qs * lds_ld(dnw, q_blk // (expert_inter // 128)),
                            )
            if const_expr(S <= 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (
                        k // (MOE_SLOTS * I_PER_SLOT) * MOE_SLOTS
                        + (k // I_PER_SLOT) % MOE_SLOTS
                    )
                    * expert_inter
                    + (k % I_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
                mark=("down", t),
            )
            if const_expr(_DN_MID8):
                # Threads past the end re-read the last word / scale and store
                # it again to the same LDS word, so they need no branch.
                N_W8 = DN_BLK * 32
                w8_ix = [
                    fx.min(tid + i * THREADS, N_W8 - 1)
                    for i in range((N_W8 + THREADS - 1) // THREADS)
                ]
                s8_ix = [
                    fx.min(tid + i * THREADS, DN_BLK - 1)
                    for i in range((DN_BLK + THREADS - 1) // THREADS)
                ]
                got8 = poll(
                    [(mb("mid8"), i, 1) for i in w8_ix]
                    + [(mb("mid8s"), i, 1) for i in s8_ix]
                )
                stamp("down", t, 2)
                for i in range_constexpr(len(w8_ix)):
                    lds_st(
                        xs,
                        w8_ix[i]
                        if const_expr(aiter_fp8_experts)
                        else f8_word(w8_ix[i] * 4),
                        got8[i][0].bitcast(fx.Float32),
                    )
                for i in range_constexpr(len(s8_ix)):
                    lds_st(misc, s8_ix[i], got8[len(w8_ix) + i][0].bitcast(fx.Float32))
            elif const_expr(GFX942):
                mid_blks = [
                    fx.min(wave + b * WAVES, DN_BLK - 1)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
                mids = get2_many([(mb("mid"), k * 128 + lane * 2) for k in mid_blks])
                stamp("down", t, 2)
            else:
                mids = get2_many(
                    [
                        (
                            mb("mid"),
                            fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2,
                        )
                        for b in range((DN_BLK + WAVES - 1) // WAVES)
                    ]
                )
                stamp("down", t, 2)
            if const_expr(_DN_QUANT_BATCH and not expert_mxfp4 and not _DN_MID8):
                # A wave's blocks past DN_BLK are clamped onto DN_BLK - 1: they
                # rewrite the same bytes as its owner, so they need no branch.
                qz = quant_scaled_many(mids)
                st_f8_many(
                    [(k * 128 + lane * 2, q[0], q[1]) for k, q in zip(mid_blks, qz)]
                )
                dws = [lds_ld(dnw, k // (expert_inter // 128)) for k in mid_blks]
                if lane == 0:
                    for b in range_constexpr(len(mid_blks)):
                        lds_st(misc, mid_blks[b], qz[b][2] * dws[b])
            for b in range_constexpr(
                0
                if _DN_QUANT_BATCH and not expert_mxfp4
                else (DN_BLK + WAVES - 1) // WAVES
            ):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    if const_expr(expert_mxfp4):
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(mids[b][0], mids[b][1])
                            st_mxfp8(blk * 128 + lane * 2, q0, q1)
                            if lane % 16 == 0:
                                lds_st(
                                    misc,
                                    blk * 4 + lane // 16,
                                    qs.bitcast(fx.Float32),
                                )
                        else:
                            lds_st(
                                xs,
                                blk * 64 + lane,
                                bf16_pair(mids[b][0], mids[b][1]),
                            )
                    else:
                        q0, q1, qs = quant_scaled(mids[b][0], mids[b][1])
                        st_f8(blk * 128 + lane * 2, q0, q1)
                        if lane == 0:
                            lds_st(
                                misc,
                                blk,
                                qs * lds_ld(dnw, blk // (expert_inter // 128)),
                            )
            stamp("down", t, 7)  # this wave's mid staged, before the barrier
            gpu.barrier()
            stamp("down", t, 5)  # mid quantized into LDS
            if const_expr(S > 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            # The shared expert leads every column's run, as it did when it was
            # slot 0 of each sample -- the condition that keeps this bit-exact.
            #
            # Built and consumed ``DN_BATCH`` at a time, which is not a style
            # choice: constructing all ``DN_NKC // 2`` of them up front holds
            # that many weight vectors live at once and cost +18 VGPRs at S=12,
            # taking the kernel to the 256 ceiling and spilling (36 B of
            # scratch a lane, 8 VGPR spills, against 0 for the old schedule).
            # Batching bounds the live set to what the routed stream already
            # keeps in flight. Order is unchanged, so this stays bit-exact.
            acc0 = None
            if const_expr(DN_SHARED):
                acc0 = [fx.Float32(0.0) for _ in range(4)]
                for c0 in range_constexpr(0, DN_NKC // 2, DN_BATCH):
                    acc0 = mma_units(
                        acc0,
                        [
                            u_dn_shared(cc)
                            for cc in range(c0, min(c0 + DN_BATCH, DN_NKC // 2))
                        ],
                    )
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre, acc0=acc0)
            stamp("down", t, 6)  # MFMA chain done

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            stamp("down", t, 3)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            def residual_a(s, row):
                return bf2_f32(get(mb("a"), (s * HIDDEN + row) // 2))

            peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)

    @flyc.jit
    def launch(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ml4_monokernel(
            h_in,
            x_out,
            cur_pos,
            positions,
            slot_mapping,
            sparse_kv_indptr,
            kv_cache,
            pe_cache,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_uk,
            s_uk,
            w_uv,
            s_uv,
            w_o,
            s_o,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch
