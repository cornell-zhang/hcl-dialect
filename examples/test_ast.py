import mlir
from mlir.ir import *
from mlir.dialects import builtin, std
from hcl_mlir import *

with Context() as ctx, Location.unknown() as loc:
    module = Module.create()

    A = placeholder((1024, 1024), name="A", ip=InsertionPoint(module.body))
    B = placeholder((1024, 1024), name="B", ip=InsertionPoint(module.body))
    C = placeholder((1024, 1024), name="C", ip=InsertionPoint(module.body))
    k = reduce_axis(0, 1024, "k")
    z = reduce_axis(0, 1024, "z")

    # func = lambda i, j: i + j
    # func = lambda i, j: i + j - j * i
    # func = lambda i, j: A[i, j]
    # func = lambda i, j: A[i, j] * B[j, i]
    func = lambda i, j: sum(A[i, j] * B[j, i], axis=k)
    # func = lambda i, j: sum(A[i, k] * B[k, j], axis=k)
    # func = lambda i, j: A[1, 0]

    for tensor in [A, B, C]:
        tensor.op = tensor.op(tensor.memref_type, None, None, None, ip=InsertionPoint(module.body))

    li = make_constant_for(0, 1024, step=1, name="i", ip=InsertionPoint(module.body))
    lj = make_constant_for(0, 1024, step=1, name="j", ip=InsertionPoint(li.body))
    ip = InsertionPoint(lj.body)
    iter_var = [IterVar(li.induction_variable), IterVar(lj.induction_variable)]
    ret = func(*iter_var)

    set_insertion_point(ip)
    builder = ASTBuilder()
    builder.visit(ret)

    module.dump()