from mlir.ir import *
from mlir.dialects import builtin
import mlir.dialects.std as std

with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get((1024, 1024), f32)

    with InsertionPoint(module.body), Location.unknown():

        @builtin.FuncOp.from_py_func(tensor_type, tensor_type, tensor_type)
        def gemm(A, B, C):
            return C

        # func = builtin.FuncOp(
        #     name="gemm",
        #     type=FunctionType.get(
        #         inputs=[tensor_type, tensor_type],
        #         results=[tensor_type])
        #     )

        # with InsertionPoint(func.add_entry_block()):
        #     std.ReturnOp([func.entry_block.arguments[0]])

    module.dump()