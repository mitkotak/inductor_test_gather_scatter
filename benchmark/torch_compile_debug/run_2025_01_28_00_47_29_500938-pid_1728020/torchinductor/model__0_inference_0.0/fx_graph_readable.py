class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 200000]", arg1_1: "f32[10000, 32]"):
         # File: /home/mkotak/atomic_architects/projects/inductor_test_gather_scatter/benchmark/test_compile_basic.py:17 in gather_scatter, code: row, col = edge_index
        select: "i64[200000]" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1: "i64[200000]" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        
         # File: /home/mkotak/atomic_architects/projects/inductor_test_gather_scatter/benchmark/test_compile_basic.py:18 in gather_scatter, code: x_j = x[row]
        index: "f32[200000, 32]" = torch.ops.aten.index.Tensor(arg1_1, [select]);  arg1_1 = select = None
        
         # File: /home/mkotak/atomic_architects/projects/inductor_test_gather_scatter/.venv/lib/python3.11/site-packages/torch_geometric/utils/_scatter.py:195 in broadcast, code: return src.view(size).expand_as(ref)
        view: "i64[200000, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[200000, 32]" = torch.ops.aten.expand.default(view, [200000, 32]);  view = None
        
         # File: /home/mkotak/atomic_architects/projects/inductor_test_gather_scatter/.venv/lib/python3.11/site-packages/torch_geometric/utils/_scatter.py:75 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        full_default: "f32[10000, 32]" = torch.ops.aten.full.default([10000, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add: "f32[10000, 32]" = torch.ops.aten.scatter_add.default(full_default, 0, expand, index);  full_default = expand = index = None
        return (scatter_add,)
        