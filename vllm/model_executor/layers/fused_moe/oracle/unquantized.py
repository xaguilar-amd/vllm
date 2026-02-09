# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe import (
    is_supported_config_trtllm_bf16,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutlass_fused_moe

logger = init_logger(__name__)


class UnquantizedMoeBackend(Enum):
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    AITER = "ROCm AITER"
    AITER_MORI_EP = "ROCm AITER + MORI EP"
    TRITON = "TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"


# NOTE(zyongye): Unsupported backend means backend
# that is not conform with Modular kernel format.
# We will directly call the kernel for those backend
UNSUPPORTED_BACKEND = [
    UnquantizedMoeBackend.FLASHINFER_TRTLLM,
    UnquantizedMoeBackend.CPU,
    UnquantizedMoeBackend.TPU,
    UnquantizedMoeBackend.OOT,
]


def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
    use_ep: bool,
    use_dp: bool,
) -> UnquantizedMoeBackend:
    """
    Select the primary Unquantized MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    def _make_log_backend(backend: UnquantizedMoeBackend):
        return f"Using {backend.value} backend for Unquantized MoE"

    rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    # Check if FlashInfer TRTLLM BF16 MoE is supported
    trtllm_supported, _ = is_supported_config_trtllm_bf16(
        moe_config=moe_config,
        activation_format=activation_format,
    )
    flashinfer_trtllm_moe_enabled = (
        has_flashinfer() and envs.VLLM_USE_FLASHINFER_MOE_FP16 and trtllm_supported
    )
    # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
    flashinfer_cutlass_moe_enabled = (
        has_flashinfer_cutlass_fused_moe()
        and envs.VLLM_USE_FLASHINFER_MOE_FP16
        and use_ep
        and (not use_dp)
        and current_platform.has_device_capability(90)
    )
    if current_platform.is_rocm():
        if rocm_aiter_moe_enabled:
            backend = UnquantizedMoeBackend.AITER
        else:
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_cuda():
        if flashinfer_trtllm_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
        elif flashinfer_cutlass_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
        else:
            if not envs.VLLM_USE_FLASHINFER_MOE_FP16 and trtllm_supported:
                logger.info_once(
                    "FlashInfer TRTLLM MoE is available but not enabled, "
                    "consider setting VLLM_USE_FLASHINFER_MOE_FP16=1 "
                    "to enable it for better performance.",
                    scope="local",
                )
            elif use_ep and (not use_dp):
                logger.info_once(
                    "FlashInfer MoE is available for EP"
                    " but not enabled, consider setting"
                    " VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.",
                    scope="local",
                )
            elif use_dp:
                logger.info_once(
                    "FlashInfer CUTLASS MoE is currently not available for DP.",
                    scope="local",
                )
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_xpu():
        backend = UnquantizedMoeBackend.XPU
    if current_platform.is_cpu():
        backend = UnquantizedMoeBackend.CPU
    if current_platform.is_tpu():
        backend = UnquantizedMoeBackend.TPU
    if current_platform.is_out_of_tree():
        backend = UnquantizedMoeBackend.OOT

    logger.info_once(_make_log_backend(backend), scope="local")
    return backend


def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unquantized_backend in (
        UnquantizedMoeBackend.AITER,
        UnquantizedMoeBackend.AITER_MORI_EP,
    ):
        # Both AITER and AITER_MORI_EP use the same weight format
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(
            layer.w13_weight.data, layer.w2_weight.data
        )

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        # Swap halves to arrange as [w3; w1] (kernel expectation)
        w13_weight = swap_w13_to_w31(layer.w13_weight.data)

    return w13_weight, w2_weight


def make_unquantized_moe_kernel(
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> mk.FusedMoEModularKernel | None:
    if backend in UNSUPPORTED_BACKEND:
        return None

    if backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=False,
        )

    elif backend == UnquantizedMoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            AiterExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
    elif backend == UnquantizedMoeBackend.AITER_MORI_EP:
        # MORI-EP + AITER: True Expert Parallelism on AMD MI300X
        # - MORI: Optimized All-to-All dispatch/combine for token routing
        # - AITER: High-performance expert computation kernels
        from vllm.model_executor.layers.fused_moe.mori_prepare_finalize import (
            MoriPrepareAndFinalize,
            is_mori_ep_available,
        )
        from vllm.model_executor.layers.fused_moe.mori_utils import (
            MoriEpConfig,
            compute_num_local_experts,
            compute_rank_expert_offset,
            create_mori_ep_op,
        )
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        if not is_mori_ep_available():
            # Fallback: MORI not installed, use AITER without EP
            logger.warning(
                "MORI-EP not available, falling back to AITER without EP. "
                "Install MORI from https://github.com/ROCm/mori for EP support."
            )
            kernel = mk.FusedMoEModularKernel(
                MoEPrepareAndFinalizeNoEP(),
                AiterExperts(
                    moe_config=moe_config,
                    quant_config=quant_config,
                ),
                inplace=not moe_config.disable_inplace,
            )
        else:
            # Calculate Expert Parallelism parameters
            ep_size = moe_config.moe_parallel_config.ep_size
            ep_rank = moe_config.moe_parallel_config.ep_rank
            num_local_experts = compute_num_local_experts(
                moe_config.num_experts, ep_size
            )
            rank_expert_offset = compute_rank_expert_offset(
                ep_rank, num_local_experts
            )

            # Create MORI EP operator (cached across all MoE layers)
            mori_config = MoriEpConfig(
                rank=ep_rank,
                world_size=ep_size,
                hidden_dim=moe_config.hidden_dim,
                max_num_tokens=moe_config.max_num_tokens,
                num_experts=moe_config.num_experts,
                topk=moe_config.experts_per_token,
                dtype=moe_config.in_dtype,
            )
            mori_ep_op = create_mori_ep_op(mori_config)

            kernel = mk.FusedMoEModularKernel(
                MoriPrepareAndFinalize(
                    ep_op=mori_ep_op,
                    num_local_experts=num_local_experts,
                    rank_expert_offset=rank_expert_offset,
                    ep_size=ep_size,
                    num_experts=moe_config.num_experts,
                    dp_size=moe_config.moe_parallel_config.dp_size,
                ),
                AiterExperts(
                    moe_config=moe_config,
                    quant_config=quant_config,
                ),
                moe_parallel_config=moe_config.moe_parallel_config,
            )
    elif backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe import TritonExperts

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
    elif backend == UnquantizedMoeBackend.XPU:
        from vllm.model_executor.layers.fused_moe import XPUExperts

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            XPUExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
    return kernel
