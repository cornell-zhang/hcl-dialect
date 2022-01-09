import mlir
from mlir.ir import *
from mlir.dialects import builtin, std
from hcl_mlir import *

with Context() as ctx, Location.unknown() as loc:
    module = Module.create()

    A = placeholder((1024, 1024), name="A", ip=InsertionPoint(module.body))
    B = placeholder((1024, 1024), name="B", ip=InsertionPoint(module.body))

    func = lambda i, j: A[i, j] + B[i, j]

    li = make_constant_for(0, 1024, step=1, name="i", ip=InsertionPoint(module.body))
    lj = make_constant_for(0, 1024, step=1, name="j", ip=InsertionPoint(li.body))
    ip = InsertionPoint(lj.body)
    iter_var = [IterVar(li.induction_variable, ip), IterVar(lj.induction_variable, ip)]
    ret = func(*iter_var)
    module.dump()