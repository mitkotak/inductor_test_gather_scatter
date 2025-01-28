
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA RTX A5500 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        select = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1 = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        index = torch.ops.aten.index.Tensor(arg1_1, [select]);  arg1_1 = select = None
        view = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand = torch.ops.aten.expand.default(view, [200000, 32]);  view = None
        full_default = torch.ops.aten.full.default([10000, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default, 0, expand, index);  full_default = expand = index = None
        return (scatter_add,)
        
def load_args(reader):
    buf0 = reader.storage(None, 3200000, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 200000), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1280000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (10000, 32), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)