# RUN: %PYTHON %s

from hcl_mlir.build_ir import IterVar
from hcl_mlir.ir import *
from hcl_mlir.dialects import builtin, arith, memref, affine
from hcl_mlir.dialects import hcl as hcl_d
import hcl_mlir

with Context() as ctx, Location.unknown() as loc:
    hcl_d.register_dialect(ctx)
    module = Module.create()
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)
    idx_type = IndexType.get()
    memref_type = MemRefType.get((1024,), f32)
    hcl_mlir.enable_build_inplace()

    with InsertionPoint(module.body):

        @builtin.FuncOp.from_py_func(memref_type)
        def kernel(A):
            for_i = hcl_mlir.make_affine_for(0, 1024, name="i")
            hcl_mlir.GlobalInsertionPoint.save(InsertionPoint(for_i.body))
            var = IterVar(for_i.induction_variable)
            var.dtype = idx_type
            with InsertionPoint(for_i.body):

                def make_if_block(arg):
                    a = memref.LoadOp(A, [for_i.induction_variable])
                    cst = hcl_mlir.ConstantOp(idx_type, 0)
                    cmp = hcl_mlir.CmpOp(var, cst, arg)
                    if_op = hcl_mlir.make_if(cmp)
                    with InsertionPoint(if_op.then_block):
                        add = arith.AddFOp(a.result, a.result)
                        affine.AffineYieldOp([])

                make_if_block("eq")
                # make_if_block("ne")
                make_if_block("lt")
                make_if_block("le")
                make_if_block("gt")
                make_if_block("ge")
                affine.AffineYieldOp([])

            return A

    module.dump()
    Module.parse(str(module))
    print("Built done!")