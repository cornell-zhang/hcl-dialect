from mlir.ir import *
from mlir.dialects import builtin, std, tensor, memref
import hcl_mlir.affine as affine

def makeConstantFor(lb, ub, step=1, name=""):
    # Construct lower bound
    lbCst = AffineConstantExpr.get(lb)
    lbMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[lbCst])
    lbMapAttr = AffineMapAttr.get(lbMap)

    # Construct upper bound
    ubCst = AffineConstantExpr.get(ub)
    ubMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[ubCst])
    ubMapAttr = AffineMapAttr.get(ubMap)

    # Construct step
    step = IntegerAttr.get(i32, step)

    # Create AffineForOp
    forOp = affine.AffineForOp(None, None, step, lbMapAttr, ubMapAttr, name=StringAttr.get(name))
    return forOp

with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)
    tensor_type = RankedTensorType.get((1024, 1024), f32)
    memref_type = MemRefType.get((1024, 1024), f32)

    with InsertionPoint(module.body):

        @builtin.FuncOp.from_py_func(memref_type, memref_type, memref_type)
        def gemm(A, B, C):
            for_i = makeConstantFor(0, 1024, name="i")
            with InsertionPoint(for_i.body):
                for_j = makeConstantFor(0, 1024, name="j")
                with InsertionPoint(for_j.body):
                    for_k = makeConstantFor(0, 1024, name="k")
                    with InsertionPoint(for_k.body):
                        a = memref.LoadOp(f32, A, [for_i.induction_variable, for_k.induction_variable])
                        b = memref.LoadOp(f32, B, [for_k.induction_variable, for_j.induction_variable])
                        c = memref.LoadOp(f32, C, [for_i.induction_variable, for_j.induction_variable])
                        prod = std.MulFOp(f32, a.result, b.result)
                        sum_ = std.AddFOp(f32, prod.result, c.result)
                        memref.StoreOp(sum_.result, C, [for_i.induction_variable, for_j.induction_variable])

        @builtin.FuncOp.from_py_func(tensor_type, tensor_type, tensor_type)
        def gemm_memref(A, B, C):
            for_i = makeConstantFor(0, 1024, name="i")
            with InsertionPoint(for_i.body):
                for_j = makeConstantFor(0, 1024, name="j")
                with InsertionPoint(for_j.body):
                    for_k = makeConstantFor(0, 1024, name="k")
                    with InsertionPoint(for_k.body):
                        a = tensor.ExtractOp(f32, A, [for_i.induction_variable, for_k.induction_variable])
                        b = tensor.ExtractOp(f32, B, [for_k.induction_variable, for_j.induction_variable])
                        c = tensor.ExtractOp(f32, C, [for_i.induction_variable, for_j.induction_variable])
                        prod = std.MulFOp(f32, a.result, b.result)
                        sum_ = std.AddFOp(f32, prod.result, c.result)
                        tensor.InsertOp(f32, sum_.result, C, [for_i.induction_variable, for_j.induction_variable])

            return C

    module.dump()