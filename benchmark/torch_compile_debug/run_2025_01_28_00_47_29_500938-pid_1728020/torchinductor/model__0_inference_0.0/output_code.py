# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_mkotak/he/cheyrsb4v7sws4l3zsxmajhmjcz52wi74jbzmwxn2ywv67x242xl.py
# Topologically Sorted Source Nodes: [new_zeros, x_j, scatter_add_], Original ATen: [aten.new_zeros, aten.index, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros => full_default
#   scatter_add_ => scatter_add
#   x_j => index
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([10000, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%select]), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %index), kwargs = {})
triton_poi_fused_index_new_zeros_scatter_add_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_new_zeros_scatter_add_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_mkotak/ob/cobiiazeux7rzi77hvccn25pjfz22bjwwzleeaimvd3cg2q53erw.py
# Topologically Sorted Source Nodes: [new_zeros, x_j, scatter_add_], Original ATen: [aten.new_zeros, aten.index, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros => full_default
#   scatter_add_ => scatter_add
#   x_j => index
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([10000, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%select]), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %index), kwargs = {})
triton_poi_fused_index_new_zeros_scatter_add_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_new_zeros_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tl.device_assert((0 <= tmp0) & (tmp0 < 10000), "index out of bounds: 0 <= tmp0 < 10000")
    tmp3 = tl.full([XBLOCK], 10000, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 10000), "index out of bounds: 0 <= tmp6 < 10000")
    tmp8 = tl.load(in_ptr1 + (x0 + (32*tmp6)), None)
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp0)), tmp8, None, sem='relaxed')
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 200000), (200000, 1))
    assert_size_stride(arg1_1, (10000, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10000, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros, x_j, scatter_add_], Original ATen: [aten.new_zeros, aten.index, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_new_zeros_scatter_add_0.run(buf0, 320000, grid=grid(320000), stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros, x_j, scatter_add_], Original ATen: [aten.new_zeros, aten.index, aten.scatter_add]
        triton_poi_fused_index_new_zeros_scatter_add_1.run(arg0_1, arg1_1, buf0, 6400000, grid=grid(6400000), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 200000), (200000, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((10000, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
