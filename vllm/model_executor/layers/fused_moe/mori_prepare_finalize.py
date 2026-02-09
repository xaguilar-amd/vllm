# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MORI-EP Prepare and Finalize implementation for Expert Parallelism.

This module provides the MoriPrepareAndFinalize class that uses MORI-EP
dispatch/combine APIs for All-to-All communication in MoE layers.
Designed to pair with AiterExperts for maximum AMD performance on MI300X.

Architecture Overview:
======================
In Expert Parallelism (EP), experts are distributed across GPUs:
- EP8 with 256 experts -> 32 experts per GPU (experts 0-31 on GPU0, etc.)
- Tokens must be routed to the GPU that owns the selected expert

The MoE forward pass with MORI-EP:
1. DISPATCH: Send tokens to GPUs owning selected experts (All-to-All)
2. COMPUTE:  Each GPU runs AITER kernels on its LOCAL experts only
3. COMBINE:  Return expert outputs to original token owners (All-to-All)

Data Flow Example (DeepSeek-R1, EP8):
====================================
- Input: [batch=128, hidden=7168] tokens on each GPU
- Router selects topk=8 experts per token (global IDs 0-255)
- Dispatch routes tokens to expert-owning GPUs
- Each GPU receives ~128x8/8 = 128 tokens (on average)
- AITER computes weighted sum of LOCAL expert outputs
- Combine returns results -> [batch=128, hidden=7168] output per GPU

MORI-EP Quantization Strategies:
================================
- Strategy A (FP8 dispatch): Quantize before dispatch for 2x bandwidth
  savings. Use when VLLM_MORI_EP_USE_FP8_DISPATCH=1.
- Strategy B (BF16 dispatch): Dispatch BF16, quantize after receive.
  Default mode, simpler but uses more bandwidth.

MORI-EP Configuration:
======================
- EP8, EP16, EP32 configurations supported
- Kernel types: IntraNode (XGMI), InterNode (RDMA)
- Requires MORI_SHMEM_HEAP_SIZE environment variable for large models

Performance (8x MI300X, XGMI):
==============================
- EP8 Dispatch: 307 GB/s, ~35us latency (128 tokens)
- EP8 Combine: 330 GB/s, ~47us latency (128 tokens)
- Communication ~10-15% overhead at high batch sizes

Reference: https://github.com/ROCm/mori
"""
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

# DBO (Disaggregated Batched Operations) support for microbatching
try:
    from vllm.v1.worker.ubatching import (
        dbo_current_ubatch_id,
        dbo_enabled,
    )
    DBO_AVAILABLE = True
except ImportError:
    DBO_AVAILABLE = False

    def dbo_current_ubatch_id() -> int:
        return 0

    def dbo_enabled() -> bool:
        return False

if TYPE_CHECKING:
    from mori.ops import EpDispatchCombineOp

logger = init_logger(__name__)

# Try to import MORI-EP
try:
    from mori.ops import (
        EpDispatchCombineOp as _EpDispatchCombineOp,
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
    )

    MORI_EP_AVAILABLE = True
    logger.info("MORI-EP is available")
except ImportError:
    MORI_EP_AVAILABLE = False
    _EpDispatchCombineOp = None  # type: ignore
    EpDispatchCombineConfig = None  # type: ignore
    EpDispatchCombineKernelType = None  # type: ignore
    logger.warning(
        "MORI-EP is not available. Install from https://github.com/ROCm/mori"
    )

def is_mori_ep_available() -> bool:
    """Check if MORI-EP is available."""
    return MORI_EP_AVAILABLE

class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Expert Parallelism prepare/finalize using MORI dispatch/combine.
    Designed to pair with AiterExperts for maximum AMD performance.

    This class implements the FusedMoEPrepareAndFinalize interface, providing:
    - prepare_async/prepare: MORI dispatch (send tokens to expert-owning GPUs)
    - finalize_async/finalize: MORI combine (return results to token owners)

    Key Features:
    - Optimized dispatch/combine kernels for MoE token routing
    - FP8 dispatch + BF16 combine support (Strategy A)
    - BF16 dispatch + post-receive quantization (Strategy B)
    - XGMI transport (intra-node, 800 GB/s aggregate)
    - RDMA transport (inter-node, InfiniBand)
    - EP8, EP16, EP32 configurations

    Important Implementation Details:
    =================================
    - MORI returns FIXED-SIZE buffers; must slice to valid tokens!
    - Expert IDs: GLOBAL (0-255) -> LOCAL (0-31) conversion required
    - Combine uses ORIGINAL topk_ids (this rank's tokens), NOT received!
    - DBO (microbatching) support via per-ubatch metadata storage
    """

    def __init__(
        self,
        ep_op: "EpDispatchCombineOp",
        num_local_experts: int,
        rank_expert_offset: int,
        ep_size: int,
        num_experts: int,
        dp_size: int = 1,
        use_fp8_dispatch: bool | None = None,
    ):
        """
        Initialize MoriPrepareAndFinalize.

        Args:
            ep_op: MORI EpDispatchCombineOp for dispatch/combine operations.
            num_local_experts: Number of experts on this GPU
                (e.g., 32 for EP8 with 256 total experts).
            rank_expert_offset: Starting global expert ID for this rank
                (e.g., rank * num_local_experts).
            ep_size: Number of EP ranks (e.g., 8 for EP8).
            num_experts: Total number of experts globally.
            dp_size: Data parallel size (default: 1).
            use_fp8_dispatch: Whether to use FP8 quantization before dispatch
                for 2x bandwidth savings. If None, uses VLLM_MORI_EP_USE_FP8_DISPATCH.
        """
        super().__init__()
        assert MORI_EP_AVAILABLE, (
            "MORI-EP package not installed. "
            "Install from https://github.com/ROCm/mori"
        )

        self.ep_op = ep_op
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.dp_size = dp_size

        # Use environment variable if not explicitly set
        if use_fp8_dispatch is None:
            use_fp8_dispatch = envs.VLLM_MORI_EP_USE_FP8_DISPATCH
        self.use_fp8_dispatch = use_fp8_dispatch

        # Handle storage for DBO microbatching
        # Under DBO microbatching we must track one handle per
        # micro-batch to avoid races between threads.
        self.handles: list[Any] = [None, None]

        # Store dispatch metadata for combine
        self._dispatch_metadata: list[dict[str, Any]] = [{}, {}]

        logger.info(
            "Initialized MoriPrepareAndFinalize: "
            "ep_size=%d, num_local_experts=%d, rank_expert_offset=%d, "
            "num_experts=%d, use_fp8_dispatch=%s",
            ep_size,
            num_local_experts,
            rank_expert_offset,
            num_experts,
            use_fp8_dispatch,
        )

    # -------------------------------------------------------------------------
    # REQUIRED PROPERTIES
    # -------------------------------------------------------------------------

    def output_is_reduced(self) -> bool:
        """MORI combine produces fully reduced output."""
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        """AITER uses Standard format [N, H], not BatchedExperts."""
        return mk.FusedMoEActivationFormat.Standard

    def num_dispatchers(self) -> int:
        """Return the number of EP ranks."""
        return self.ep_size

    def max_num_tokens_per_rank(self) -> int | None:
        """No fixed limit on tokens per rank."""
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        """MORI expects int32 for topk indices."""
        return torch.int32

    def supports_async(self) -> bool:
        """MORI supports async dispatch for compute/comm overlap."""
        return True

    # -------------------------------------------------------------------------
    # PREPARE: Dispatch tokens to expert owners
    # -------------------------------------------------------------------------

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
    ) -> Callable[[], mk.PrepareResultType]:
        """
        Internal dispatch implementation.

        Args:
            tokens: Input tokens (may be FP8 quantized or BF16)
            token_scales: Scales if tokens are FP8 quantized
            topk_ids: Selected expert IDs (global)
            topk_weights: Router weights
            num_experts: Total number of experts
            a1_scale: Activation scale for post-dispatch quantization
            quant_config: Quantization configuration

        Returns:
            Callable that returns PrepareResultType when invoked
        """
        has_scales = token_scales is not None

        # MORI dispatch - send tokens to expert owners
        dispatch_result = self.ep_op.dispatch(
            input=tokens,
            weights=topk_weights,
            scales=token_scales,
            indices=topk_ids.to(torch.int32),
        )

        # Record the handle/metadata for this ubatch
        ubatch_idx = dbo_current_ubatch_id()
        self._dispatch_metadata[ubatch_idx] = {
            "dispatch_result": dispatch_result,
            "original_topk_ids": topk_ids,
            "original_topk_weights": topk_weights,
        }

        return lambda: self._receiver(
            dispatch_result=dispatch_result,
            has_scales=has_scales,
            num_experts=num_experts,
            a1_scale=a1_scale,
            quant_config=quant_config,
        )

    def _receiver(
        self,
        dispatch_result: tuple,
        has_scales: bool,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """
        Process dispatch results and prepare for expert computation.

        This is the core of the dispatch phase, executing AFTER MORI has
        completed the All-to-All communication. The function:

        1. Unpacks MORI dispatch results (fixed-size buffers)
        2. Slices buffers to valid tokens only (CRITICAL for correctness!)
        3. Converts global expert IDs (0-255) to local IDs (0-31)
        4. Masks weights for non-local experts to zero
        5. Applies post-dispatch quantization if using Strategy B
        """
        # Unpack dispatch result
        # MORI returns FIXED-SIZE buffers [max_num_tokens, ...] (e.g., [8192, 7168])
        # Only positions 0..total_recv_tokens-1 contain valid data!
        recv_x = dispatch_result[0]
        recv_weights = dispatch_result[1] if len(dispatch_result) > 1 else None
        recv_scale = dispatch_result[2] if len(dispatch_result) > 2 else None
        recv_topk_ids = dispatch_result[3] if len(dispatch_result) > 3 else None
        total_recv_tokens = dispatch_result[4] if len(dispatch_result) > 4 else None

        if has_scales:
            expert_x = recv_x
            expert_x_scale = recv_scale
        else:
            expert_x = recv_x
            expert_x_scale = None

        # CRITICAL: Slice buffers to only valid tokens!
        # MORI returns fixed-size buffers [max_num_tokens, ...] but only
        # positions 0..total_recv_tokens-1 contain valid data.
        # Passing garbage data to expert computation corrupts output.
        #
        # NOTE: This uses .item() which breaks CUDA graph capture.
        # For CUDA graph support, we'll need a different approach.
        num_valid = None
        if total_recv_tokens is not None:
            num_valid = int(total_recv_tokens.item())
            if num_valid < expert_x.shape[0]:
                expert_x = expert_x[:num_valid]
                if expert_x_scale is not None:
                    expert_x_scale = expert_x_scale[:num_valid]
                if recv_weights is not None:
                    recv_weights = recv_weights[:num_valid]
                if recv_topk_ids is not None:
                    recv_topk_ids = recv_topk_ids[:num_valid]

        # Expert ID handling: Convert GLOBAL IDs to LOCAL IDs
        #
        # MORI dispatch returns GLOBAL expert IDs (0-255), same as router output.
        # But after MORI dispatch, each rank ONLY has tokens for its local experts.
        # We need LOCAL IDs (0-31) for AITER when expert_map=None.
        #
        # Conversion: local_id = global_id - rank_expert_offset
        # Example: Rank 2 (offset=64), global ID 70 -> local ID 6
        #
        # IMPORTANT: recv_topk_ids has shape [N_recv, topk] with ALL original
        # expert IDs. But after MORI routing, only IDs for THIS rank are valid.
        # We convert all IDs to local and let AITER use expert_map to filter.
        if recv_topk_ids is not None:
            # Convert GLOBAL IDs to LOCAL IDs and zero-out non-local expert weights
            global_topk_ids = recv_topk_ids.to(torch.int64)

            # Convert to local IDs
            local_topk_ids = global_topk_ids - self.rank_expert_offset

            # Create mask for local experts
            is_local_expert = (local_topk_ids >= 0) & (local_topk_ids < self.num_local_experts)

            # Zero out weights for non-local experts
            if recv_weights is not None:
                recv_weights = recv_weights * is_local_expert.float()

            # Clamp local IDs to valid range (non-local will have 0 weight anyway)
            expert_topk_ids = local_topk_ids.clamp(min=0, max=self.num_local_experts - 1)
        else:
            expert_topk_ids = None

        # Strategy B: BF16 dispatch, quantize after receive
        # MORI kernels support block-quantized dispatch (Strategy A)
        # For non-block quantization, we dispatch BF16 and quantize here
        if not quant_config.is_block_quantized:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                )

        # Create ExpertTokensMetadata from MORI's token distribution
        # AITER computes this internally, so we pass None
        expert_tokens_meta = None

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            recv_weights,
        )

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        """
        Async prepare: Dispatch tokens to GPUs that own selected experts.

        WITH MORI: expert_map is NOT used - MORI handles routing internally
        WITHOUT MORI: expert_map would be used to remap global->local expert IDs
        """
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support "
                "defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        # Step 1: Optional weight application (for topk=1 models)
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Step 2: Quantization strategy selection
        if quant_config.is_block_quantized and self.use_fp8_dispatch:
            # Strategy A: FP8 dispatch - Quantize before dispatch for 2x BW savings
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            # Strategy B: BF16 dispatch - Dispatch BF16, quantize after receive
            a1q = a1
            a1q_scale = None
            a1_post_scale = quant_config.a1_scale

        # Step 3: Execute dispatch
        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Synchronous prepare - calls async version and waits."""
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
        )
        return receiver()

    # -------------------------------------------------------------------------
    # FINALIZE: Combine results back to original token owners
    # -------------------------------------------------------------------------

    def _finalize_impl(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        """
        Internal finalize implementation - executes MORI combine phase.

        This function completes the MoE layer by:
        1. Applying weight reduction (if needed) to expert outputs
        2. Calling MORI combine to route results back to original token owners
        3. Slicing the fixed-size combine output to actual batch size
        4. Copying combined results to output buffer

        CRITICAL: Combine uses ORIGINAL topk_ids!
        =========================================
        The topk_ids passed to combine must be the ORIGINAL indices from
        THIS rank's tokens, NOT the received indices from dispatch!
        """
        ubatch_idx = dbo_current_ubatch_id()

        # Retrieve ORIGINAL topk_ids from dispatch metadata
        # CRITICAL: Combine needs the ORIGINAL indices [M, 8] (this rank's tokens)
        # to know what results to receive, NOT the received indices [N_recv, 8]!
        dispatch_meta = self._dispatch_metadata[ubatch_idx]
        original_topk_ids = dispatch_meta.get("original_topk_ids", topk_ids)

        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        if fused_expert_output.numel() != 0:
            # Apply weights before combine if using delegate
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # MORI combine expects BF16
        assert fused_expert_output.dtype == torch.bfloat16, (
            f"Expected fused_expert_output bfloat16, got {fused_expert_output.dtype}"
        )

        # MORI combine - returns results to original token owners
        # NOTE: We pass weights=None because AITER fused_moe already applies
        # topk_weights during expert computation.
        # CRITICAL: Use ORIGINAL topk_ids (this rank's tokens), NOT received!
        combine_result = self.ep_op.combine(
            input=fused_expert_output,
            weights=None,  # AITER already applied weights
            indices=original_topk_ids.to(torch.int32),
            call_reset=True,  # Reset for next iteration
        )

        combined_x = combine_result[0]

        # MORI combine returns a fixed-size buffer [max_num_tokens, hidden_dim]
        # but the actual batch may be smaller. Slice to match output shape.
        num_tokens = output.shape[0]
        if combined_x.shape[0] != num_tokens:
            combined_x = combined_x[:num_tokens]

        # Clear dispatch metadata for this ubatch
        self._dispatch_metadata[ubatch_idx] = {}

        if do_async:
            # Capture variables for the closure
            _combined_x = combined_x
            _output = output

            def _receiver():
                # Respect inplace outputs
                _output.copy_(_combined_x, non_blocking=True)

            return _receiver
        else:
            # Synchronous: copy immediately
            output.copy_(combined_x, non_blocking=True)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        """Async finalize - returns callable that completes finalization."""
        receiver = self._finalize_impl(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """Synchronous finalize - completes immediately."""
        self._finalize_impl(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
