import os 
import ctypes
import numpy as np
import mlir.all_passes_registration

from mlir.ir import *
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.runtime import *

def lowerToLLVM(module):
    import mlir.conversions
    pm = PassManager.parse(
        "lower-affine,convert-scf-to-std,canonicalize,convert-memref-to-llvm,convert-std-to-llvm,reconcile-unrealized-casts")
    pm.run(module)
    # module.dump()
    return module

def get_assembly(filename):
    with open(filename, "r") as f:
        code = f.read()
    return code


def test_execution_engine(P=16, Q=22, R=18, S=24):
    code = get_assembly("./affine_dialect.mlir")

    # Add shared library
    shared_libs = [
        os.path.join(os.getenv("LLVM_BUILD_DIR"), 'lib', 'libmlir_runner_utils.so'),
        os.path.join(os.getenv("LLVM_BUILD_DIR"), 'lib', 'libmlir_c_runner_utils.so')
    ]

    A = np.random.randint(10, size=(P, Q)).astype(np.float32)
    B = np.random.randint(10, size=(Q, R)).astype(np.float32)
    C = np.random.randint(10, size=(R, S)).astype(np.float32)
    D = np.random.randint(10, size=(P, S)).astype(np.float32)
    # res1 = np.zeros((P, S), dtype=np.float32)

    A_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(A)))
    B_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(B)))
    C_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(C)))
    D_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(D)))
    # res1_memref = ctypes.pointer(
        # ctypes.pointer(get_ranked_memref_descriptor(res1))
    # )
    res1 = make_nd_memref_descriptor(2, ctypes.c_float)()
    res1_memref = ctypes.pointer(ctypes.pointer(res1))

    with Context():
        module = Module.parse(code)
        lowered = lowerToLLVM(module)
        execution_engine = ExecutionEngine(lowered, opt_level=3, shared_libs=shared_libs)
        execution_engine.invoke("top", res1_memref, A_memref, B_memref, C_memref, D_memref)
        
    ret = ranked_memref_to_numpy(res1_memref[0])
    print(ret)

if __name__ == "__main__":
    test_execution_engine()
