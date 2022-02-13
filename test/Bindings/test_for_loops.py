from hcl_mlir.ir import *
from hcl_mlir.dialects import builtin, arith, memref, affine
from hcl_mlir.dialects import hcl as hcl_d
import hcl_mlir

with Context() as ctx, Location.unknown() as loc:
    hcl_d.register_dialect(ctx)
    module = Module.create()
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)
    memref_type = MemRefType.get((1024, 1024), f32)

    with InsertionPoint(module.body):
        hcl_d.CreateLoopHandleOp(hcl_d.LoopHandleType.get(ctx),
                                    StringAttr.get("i"))
        hcl_d.CreateLoopHandleOp(hcl_d.LoopHandleType.get(ctx),
                                    StringAttr.get("j"))
        hcl_d.CreateLoopHandleOp(hcl_d.LoopHandleType.get(ctx),
                                    StringAttr.get("k"))

        @builtin.FuncOp.from_py_func(memref_type, memref_type, memref_type)
        def gemm(A, B, C):
            for_i = hcl_mlir.make_affine_for(0, 1024, name="i")
            with InsertionPoint(for_i.body):
                for_j = hcl_mlir.make_affine_for(0, 1024, name="j")
                with InsertionPoint(for_j.body):
                    for_k = hcl_mlir.make_affine_for(0, 1024, name="k")
                    with InsertionPoint(for_k.body):
                        a = memref.LoadOp(A, [for_i.induction_variable, for_k.induction_variable])
                        b = memref.LoadOp(B, [for_k.induction_variable, for_j.induction_variable])
                        c = memref.LoadOp(C, [for_i.induction_variable, for_j.induction_variable])
                        prod = arith.MulFOp(a.result, b.result)
                        sum_ = arith.AddFOp(prod.result, c.result)
                        memref.StoreOp(sum_.result, C, [for_i.induction_variable, for_j.induction_variable])
                        affine.AffineYieldOp([])
                    affine.AffineYieldOp([])
                affine.AffineYieldOp([])

            for_i = hcl_mlir.make_affine_for(0, 1024, name="i")
            with InsertionPoint(for_i.body):
                for_j = hcl_mlir.make_affine_for(0, 1024, name="j")
                with InsertionPoint(for_j.body):
                    for_k = hcl_mlir.make_affine_for(0, 1024, name="k")
                    with InsertionPoint(for_k.body):
                        a = affine.AffineLoadOp(A, [for_i.induction_variable, for_k.induction_variable])
                        b = affine.AffineLoadOp(B, [for_k.induction_variable, for_j.induction_variable])
                        c = affine.AffineLoadOp(C, [for_i.induction_variable, for_j.induction_variable])
                        prod = arith.MulFOp(a.result, b.result)
                        sum_ = arith.AddFOp(prod.result, c.result)
                        memref.StoreOp(sum_.result, C, [for_i.induction_variable, for_j.induction_variable])
                        affine.AffineYieldOp([])
                    affine.AffineYieldOp([])
                affine.AffineYieldOp([])

            return C

    module.dump()
    Module.parse(str(module))
    print("Built done!")