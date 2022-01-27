import ctypes
import numpy as np
import os

import mlir.all_passes_registration

from mlir import ir
from mlir import runtime as rt
from mlir import execution_engine
from mlir import passmanager

from mlir.dialects import builtin

class Compiler:
    def __init__(self):
        self.pipeline = (
            f'lower-affine,' 
            f'convert-scf-to-std,' 
            f'convert-memref-to-llvm,'
            f'convert-std-to-llvm'
        )
    def __call__(self, module: ir.Module):
        passmanager.PassManager.parse(self.pipeline).run(module)


def code():
    return f"""
module  {{
memref.global "private" @gv0 : memref<2xf32> = dense<[1.0, 2.0]>
  func @top(%arg0: memref<2xf32>) -> memref<1xf32> {{
    %1 = memref.alloc() : memref<1xf32>
    affine.for %arg1 = 0 to 1 {{
      %3 = memref.alloc() : memref<1xf32>
      // FIX: initialize to zero
      %c3_0 = constant 0 : index
      %c3 = constant 0.0 : f32
      affine.store %c3, %3[%c3_0] : memref<1xf32>
      %U3 = memref.cast %3 : memref<1xf32> to memref<*xf32>
      call @print_memref_f32(%U3) : (memref<*xf32>) -> ()
      affine.for %arg2 = 0 to 2 {{
        %5 = affine.load %arg0[%arg2] : memref<2xf32>
        %c0_0 = constant 0 : index
        %6 = affine.load %3[%c0_0] : memref<1xf32>
        %7 = addf %5, %6 : f32
        affine.store %7, %3[%c0_0] : memref<1xf32>
      }} {{loop_name = "x"}}
      %c0 = constant 0 : index
      %4 = affine.load %3[%c0] : memref<1xf32>
      affine.store %4, %1[%arg1] : memref<1xf32>
    }} {{loop_name = "_", stage_name = "sum"}}
    return %1 : memref<1xf32>
  }}

  func @main(%0 : memref<2xf32>) -> memref<1xf32> attributes {{ llvm.emit_c_interface }} {{
    %gv0 = memref.get_global @gv0 : memref<2xf32>
    //%U0 = memref.cast %0 : memref<2xf32> to memref<*xf32>
    //call @print_memref_f32(%U0) : (memref<*xf32>) -> ()
    %1 = call @top(%gv0) : (memref<2xf32>) -> (memref<1xf32>)
    %U = memref.cast %1 : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%U) : (memref<*xf32>) -> ()
    return %1 : memref<1xf32>
  }}
  func private @print_memref_f32(memref<*xf32>) attributes {{ llvm.emit_c_interface }}
}}
"""

def main():
    support_lib = [
        "/work/shared/users/phd/nz264/llvm-13.0/build/lib/libmlir_c_runner_utils.so",
        "/work/shared/users/phd/nz264/llvm-13.0/build/lib/libmlir_runner_utils.so"
    ]

    with ir.Context() as ctx, ir.Location.unknown():
        compiler = Compiler()
        module = ir.Module.parse(code())
        compiler(module)
        engine = execution_engine.ExecutionEngine(module, opt_level=0, shared_libs=support_lib)
        a = np.array([1, 2], np.float32)
        b = np.zeros((1,), np.float32)
        mem_a  = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
        mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
        engine.invoke('main', mem_b, mem_a)
        out = rt.ranked_memref_to_numpy(mem_b[0])
        print(out)

if __name__ == "__main__":
    main()