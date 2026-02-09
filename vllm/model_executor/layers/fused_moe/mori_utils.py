# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MORI-EP Utilities: Configuration, Operator Creation, and Memory Management.

This module provides utilities for creating and managing MORI EP
(Expert Parallelism) dispatch/combine operators for MoE layers on AMD GPUs.

Key Components:
===============
1. MoriEpConfig: Dataclass holding all MORI EP configuration parameters
2. create_mori_ep_op(): Factory function to create/cache MORI operators
3. _ensure_mori_shmem_initialized(): Thread-safe shared memory initialization

Operator Caching:
=================
MORI EP operators are CACHED and SHARED across all MoE layers!
This is critical because:
- Each operator allocates symmetric heap memory
- A model like DeepSeek-R1 has 61 MoE layers
- Without caching: 61 operators would exhaust symmetric heap
- With caching: 1 operator shared by all layers

Cache key includes: rank, world_size, hidden_dim, max_tokens, topk, etc.
Same configuration → same operator reused.

Memory Requirements:
===================
MORI requires symmetric heap memory for All-to-All communication buffers.
The heap is SHARED across all GPUs (ranks).

Memory calculation:
  per_rank = max_tokens × hidden_dim × 16 bytes (BF16 + metadata)
  total = per_rank × num_ranks

Example for DeepSeek-R1 (EP8):
  per_rank = 8192 × 7168 × 16 = 940 MB
  total = 940 × 8 = 7.5 GB minimum
  recommended = 12 GB

CRITICAL: Set MORI_SHMEM_HEAP_SIZE BEFORE starting the server:
  export MORI_SHMEM_HEAP_SIZE=12G

Without this, you'll see "Out of symmetric heap memory" errors.

Initialization Sequence:
========================
1. First call to create_mori_ep_op() triggers shmem initialization
2. _ensure_mori_shmem_initialized() calls shmem_torch_process_group_init()
3. MORI integrates directly with PyTorch's process group (no manual broadcast needed)
4. Subsequent calls to create_mori_ep_op() skip shmem init (already done)

Reference: https://github.com/ROCm/mori
"""
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from mori.ops import EpDispatchCombineOp

logger = init_logger(__name__)

# Try to import MORI
try:
    from mori.ops import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp as _EpDispatchCombineOp,
    )
    from mori import shmem

    MORI_EP_AVAILABLE = True
except ImportError:
    MORI_EP_AVAILABLE = False
    EpDispatchCombineConfig = None  # type: ignore
    EpDispatchCombineKernelType = None  # type: ignore
    _EpDispatchCombineOp = None  # type: ignore
    shmem = None  # type: ignore

# Track if shmem has been initialized
_MORI_SHMEM_INITIALIZED = False
_MORI_SHMEM_LOCK = threading.Lock()

# Global cache for MORI EP operators - shared across all MoE layers
# Key is a tuple of (rank, world_size, hidden_dim, max_num_tokens, topk, kernel_type)
_MORI_EP_OP_CACHE: dict[tuple, "_EpDispatchCombineOp"] = {}
_MORI_EP_OP_CACHE_LOCK = threading.Lock()


def is_mori_ep_available() -> bool:
    """Check if MORI-EP is available."""
    return MORI_EP_AVAILABLE


@dataclass
class MoriEpConfig:
    """
    Configuration for MORI EP dispatch/combine operations.

    This dataclass holds all parameters needed to create a MORI EP operator.
    The same config will result in the same cached operator.

    Core Parameters:
    ================
    - rank: Current process rank in EP group (0 to world_size-1)
    - world_size: Total number of EP ranks (EP size, e.g., 8 for EP8)
    - hidden_dim: Model hidden dimension (e.g., 7168 for DeepSeek-R1)
    - max_num_tokens: Maximum tokens per dispatch (e.g., 8192)
    - num_experts: Total global experts (e.g., 256)
    - topk: Experts selected per token (e.g., 8)

    Quantization:
    =============
    - dtype: Token data type (default: bfloat16)
    - use_fp8_dispatch: If True, use FP8 for 2x bandwidth savings

    Kernel Selection:
    =================
    - kernel_type: Transport type
        - "IntraNode": XGMI for single-node (800 GB/s aggregate)
        - "InterNode": RDMA for multi-node (InfiniBand)
        - "InterNodeV1", "InterNodeV1LL": Alternative multi-node impls

    Performance Tuning:
    ===================
    - gpu_per_node: GPUs per node (8 for MI300X)
    - warp_num_per_block: Warps per block (default 8)
    - block_num: Thread blocks (default 80)
    - rdma_block_num: Blocks for RDMA (InterNode only)
    - num_qp_per_pe: Queue pairs per PE (RDMA tuning)

    Example:
    ========
        config = MoriEpConfig(
            rank=0,
            world_size=8,
            hidden_dim=7168,
            max_num_tokens=8192,
            num_experts=256,
            topk=8,
        )
        ep_op = create_mori_ep_op(config)
    """

    # Core configuration
    rank: int
    world_size: int  # EP size (number of EP ranks)
    hidden_dim: int  # Model hidden dimension
    max_num_tokens: int  # Max tokens per batch/dispatch
    num_experts: int  # Total experts globally
    topk: int  # Experts per token (num_experts_per_token)

    # Data types
    dtype: torch.dtype = torch.bfloat16  # Token data type
    use_fp8_dispatch: bool = False  # Use FP8 for 2x bandwidth savings

    # Kernel type (IntraNode for XGMI, InterNode for RDMA)
    kernel_type: str = "IntraNode"

    # Performance tuning parameters
    gpu_per_node: int = 8  # GPUs per node (MI300X = 8)
    warp_num_per_block: int = 8  # Warps per block
    block_num: int = 80  # Total thread blocks
    rdma_block_num: int = 0  # Blocks for RDMA (InterNode)
    num_qp_per_pe: int = 1  # Queue pairs per PE (RDMA tuning)

    @property
    def num_experts_per_rank(self) -> int:
        """
        Number of experts per EP rank.
        
        Example: 256 experts / 8 ranks = 32 experts per rank
        """
        return self.num_experts // self.world_size

    @property
    def scale_dim(self) -> int:
        """
        Scale dimension for FP8 quantization.
        
        FP8 uses per-128-element scales: hidden_dim // 128
        Returns 0 if not using FP8 dispatch.
        """
        return self.hidden_dim // 128 if self.use_fp8_dispatch else 0

    @property
    def scale_type_size(self) -> int:
        """
        Size of scale type in bytes.
        
        FP8 scales are float32 (4 bytes).
        Returns 0 if not using FP8 dispatch.
        """
        return 4 if self.use_fp8_dispatch else 0  # float32 scales

    @property
    def max_token_type_size(self) -> int:
        """
        Size of token data type in bytes.
        
        Used by MORI for buffer allocation:
        - FP8: 1 byte
        - FP16/BF16: 2 bytes
        - FP32: 4 bytes
        """
        if self.dtype == torch.float8_e4m3fn:
            return 1
        elif self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            return 2
        elif self.dtype == torch.float32:
            return 4
        else:
            return 2  # Default to BF16


def get_kernel_type(kernel_type_str: str) -> Any:
    """Convert kernel type string to MORI enum."""
    if not MORI_EP_AVAILABLE:
        return None

    mapping = {
        "IntraNode": EpDispatchCombineKernelType.IntraNode,
        "InterNode": EpDispatchCombineKernelType.InterNode,
        "InterNodeV1": EpDispatchCombineKernelType.InterNodeV1,
        "InterNodeV1LL": EpDispatchCombineKernelType.InterNodeV1LL,
    }
    return mapping.get(kernel_type_str, EpDispatchCombineKernelType.IntraNode)


def _ensure_mori_shmem_initialized() -> None:
    """
    Ensure MORI shared memory is initialized.
    
    This must be called before creating any MORI EP operators.
    Initializes MORI's shared memory layer using PyTorch process group integration.
    
    Thread-safe - uses a lock to prevent multiple initializations.
    """
    global _MORI_SHMEM_INITIALIZED
    
    # Fast path - already initialized
    if _MORI_SHMEM_INITIALIZED:
        return
    
    with _MORI_SHMEM_LOCK:
        # Double-check after acquiring lock
        if _MORI_SHMEM_INITIALIZED:
            return
        
        if not MORI_EP_AVAILABLE:
            raise RuntimeError("MORI-EP not available for shmem initialization")
        
        import torch.distributed as dist
        from vllm.distributed import get_world_group
        
        if not dist.is_initialized():
            raise RuntimeError(
                "Cannot initialize MORI shmem: PyTorch distributed not initialized"
            )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(
            "Initializing MORI shmem: rank=%d, world_size=%d", rank, world_size
        )
        
        try:
            # New MORI API: Use PyTorch process group integration
            # Get the world group name for initialization
            world_group = get_world_group()
            
            # Try to get group_name from different possible attributes
            group_name = None
            if hasattr(world_group, 'group_name'):
                group_name = world_group.group_name
            elif hasattr(world_group, 'cpu_group') and hasattr(world_group.cpu_group, 'group_name'):
                group_name = world_group.cpu_group.group_name
            elif hasattr(world_group, 'device_group') and hasattr(world_group.device_group, 'group_name'):
                group_name = world_group.device_group.group_name
            else:
                # Fallback: use "default" as group name
                group_name = "default"
            
            # Initialize shmem using PyTorch process group
            result = shmem.shmem_torch_process_group_init(group_name)
            
            if result != 0:
                raise RuntimeError(f"MORI shmem_torch_process_group_init returned error code: {result}")
            
            logger.info("MORI shmem initialized successfully on rank %d", rank)
            _MORI_SHMEM_INITIALIZED = True
            
        except Exception as e:
            raise RuntimeError(f"MORI shmem initialization failed on rank {rank}: {e}") from e


def _make_cache_key(config: MoriEpConfig) -> tuple:
    """
    Create a cache key for the MORI EP operator.
    
    Operators with the same configuration can be shared across MoE layers.
    """
    return (
        config.rank,
        config.world_size,
        config.hidden_dim,
        config.max_num_tokens,
        config.topk,
        config.num_experts_per_rank,
        config.kernel_type,
        config.dtype,
        config.use_fp8_dispatch,
    )


def create_mori_ep_op(config: MoriEpConfig) -> "EpDispatchCombineOp":
    """
    Create or retrieve a cached MORI EP dispatch/combine operator.
    
    This is the main factory function for MORI EP operators. It:
    1. Ensures MORI shmem is initialized (first call only)
    2. Checks cache for existing operator with same config
    3. Creates new operator if not cached
    4. Stores in cache for reuse by other MoE layers

    CRITICAL: Operator Caching
    ==========================
    MORI EP operators are SHARED across all MoE layers! This is essential
    because each operator allocates symmetric heap memory. A model like
    DeepSeek-R1 has 61 MoE layers - without caching, we'd exhaust the heap.

    The cache key includes: rank, world_size, hidden_dim, max_tokens, topk,
    num_experts_per_rank, kernel_type, dtype, use_fp8_dispatch.

    Same configuration → same cached operator returned.

    Memory Warning:
    ===============
    If MORI_SHMEM_HEAP_SIZE is not set, this function logs a warning with
    estimated memory requirements. Set this environment variable before
    starting the server to avoid "Out of symmetric heap memory" errors.

    Thread Safety:
    ==============
    This function is thread-safe. Uses locks for both shmem initialization
    and cache access to prevent race conditions in multi-threaded scenarios.

    Args:
        config: MoriEpConfig with all required parameters including:
            - rank, world_size: EP group configuration
            - hidden_dim, max_num_tokens: Buffer sizing
            - num_experts, topk: Expert configuration
            - kernel_type: "IntraNode" (XGMI) or "InterNode" (RDMA)

    Returns:
        EpDispatchCombineOp: MORI EP operator handle.
            May be a cached instance if same config was used before.

    Raises:
        RuntimeError: If MORI-EP is not installed or operator creation fails.

    Example:
        config = MoriEpConfig(rank=0, world_size=8, hidden_dim=7168,
                              max_num_tokens=8192, num_experts=256, topk=8)
        ep_op = create_mori_ep_op(config)
        
        # Later, same config returns cached operator:
        ep_op2 = create_mori_ep_op(config)  # Same object as ep_op
    """
    import os
    
    if not MORI_EP_AVAILABLE:
        raise RuntimeError(
            "MORI-EP not installed. Install from https://github.com/ROCm/mori"
        )

    # Check if MORI heap size is configured
    # Approximate memory needed: max_tokens * hidden_dim * 16 bytes (BF16 + metadata)
    estimated_mb = (config.max_num_tokens * config.hidden_dim * 16) // (1024 * 1024)
    heap_size_str = os.environ.get("MORI_SHMEM_HEAP_SIZE", "")
    if not heap_size_str:
        logger.warning(
            "MORI_SHMEM_HEAP_SIZE not set! Estimated memory needed: ~%d MB. "
            "Default heap (~260MB) may be too small. "
            "Set MORI_SHMEM_HEAP_SIZE=2G or larger to avoid 'Out of symmetric "
            "heap memory' errors.",
            estimated_mb,
        )

    # Ensure shmem is initialized before creating operator
    _ensure_mori_shmem_initialized()
    
    # Check cache first
    cache_key = _make_cache_key(config)
    
    with _MORI_EP_OP_CACHE_LOCK:
        if cache_key in _MORI_EP_OP_CACHE:
            logger.debug(
                "Reusing cached MORI EP operator for rank %d/%d "
                "(hidden=%d, max_tokens=%d, topk=%d)",
                config.rank,
                config.world_size,
                config.hidden_dim,
                config.max_num_tokens,
                config.topk,
            )
            return _MORI_EP_OP_CACHE[cache_key]
        
        # Not in cache - create new operator
        logger.info(
            "Creating MORI EP operator (will be shared across layers): "
            "world_size=%d, rank=%d, max_tokens=%d, hidden_dim=%d, topk=%d, "
            "num_experts_per_rank=%d, kernel_type=%s",
            config.world_size,
            config.rank,
            config.max_num_tokens,
            config.hidden_dim,
            config.topk,
            config.num_experts_per_rank,
            config.kernel_type,
        )

        try:
            # Create MORI EP configuration (new API removed num_qp_per_pe)
            mori_config = EpDispatchCombineConfig(
                data_type=config.dtype,
                rank=config.rank,
                world_size=config.world_size,
                hidden_dim=config.hidden_dim,
                scale_dim=config.scale_dim,
                scale_type_size=config.scale_type_size,
                max_token_type_size=config.max_token_type_size,
                max_num_inp_token_per_rank=config.max_num_tokens,
                num_experts_per_rank=config.num_experts_per_rank,
                num_experts_per_token=config.topk,
                warp_num_per_block=config.warp_num_per_block,
                block_num=config.block_num,
                use_external_inp_buf=True,
                kernel_type=get_kernel_type(config.kernel_type),
                gpu_per_node=config.gpu_per_node,
                rdma_block_num=config.rdma_block_num,
            )

            # Create the operator
            op = _EpDispatchCombineOp(mori_config)
            
            # Cache it for reuse by other layers
            _MORI_EP_OP_CACHE[cache_key] = op

            logger.info(
                "Successfully created and cached MORI EP operator for rank %d/%d",
                config.rank,
                config.world_size,
            )

            return op
        except Exception as e:
            raise RuntimeError(
                f"MORI EP operator creation failed on rank {config.rank}: {e}. "
                f"Config: world_size={config.world_size}, hidden_dim={config.hidden_dim}, "
                f"max_tokens={config.max_num_tokens}, kernel_type={config.kernel_type}"
            ) from e


def init_mori_shmem_from_process_group(group_name: str = "default") -> int:
    """
    Initialize MORI shmem from PyTorch process group.

    This should be called once during model initialization.

    Args:
        group_name: Name of the PyTorch distributed process group.

    Returns:
        int: Status code (0 for success).
    """
    assert MORI_EP_AVAILABLE, "MORI-EP not installed"
    return shmem.shmem_torch_process_group_init(group_name)


def mori_shmem_barrier() -> None:
    """Global barrier synchronization for MORI shmem."""
    assert MORI_EP_AVAILABLE, "MORI-EP not installed"
    shmem.shmem_barrier_all()


def get_mori_ep_config_for_model(
    model_config,
    parallel_config,
    max_num_tokens: int,
) -> MoriEpConfig:
    """
    Get MORI EP configuration parameters for a given model.

    Args:
        model_config: vLLM model configuration.
        parallel_config: vLLM parallel configuration.
        max_num_tokens: Maximum number of tokens per batch.

    Returns:
        MoriEpConfig: Configuration for create_mori_ep_op.
    """
    import torch.distributed as dist

    # Extract MoE configuration from model config
    hf_config = model_config.hf_config

    # Try to get MoE-specific parameters
    num_experts = getattr(hf_config, "n_routed_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", 256)

    topk = getattr(hf_config, "num_experts_per_tok", None)
    if topk is None:
        topk = getattr(hf_config, "moe_top_k", 8)

    hidden_size = getattr(hf_config, "hidden_size", 7168)

    # Get EP size and rank
    ep_size = parallel_config.tensor_parallel_size
    rank = dist.get_rank() if dist.is_initialized() else 0

    return MoriEpConfig(
        rank=rank,
        world_size=ep_size,
        hidden_dim=hidden_size,
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        topk=topk,
    )


def compute_num_local_experts(
    num_experts: int,
    ep_size: int,
) -> int:
    """
    Compute the number of local experts for a given EP configuration.

    Args:
        num_experts: Total number of experts.
        ep_size: Number of EP ranks.

    Returns:
        int: Number of experts per rank.
    """
    assert num_experts % ep_size == 0, (
        f"Number of experts ({num_experts}) must be divisible by "
        f"EP size ({ep_size})"
    )
    return num_experts // ep_size


def compute_rank_expert_offset(
    ep_rank: int,
    num_local_experts: int,
) -> int:
    """
    Compute the starting global expert ID for a given rank.

    Args:
        ep_rank: Current EP rank.
        num_local_experts: Number of experts per rank.

    Returns:
        int: Starting global expert ID for this rank.
    """
    return ep_rank * num_local_experts
