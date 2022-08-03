// RUN: hcl-opt %s --affine-to-gpu

// This is an example of gemm on GPU.
// The affine memrory accesses are manually converted to memref load/store.
// The reduction loop's result memref is manually promoted outside the loop nest.

module {
  func private @print_memref_f32(memref<*xf32>)
  func @main() {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x32xf32>
    %1 = memref.alloc() : memref<32x32xf32>
    %2 = memref.alloc() {name = "C"} : memref<32x32xf32>
    %3 = memref.cast %0 : memref<32x32xf32> to memref<*xf32>
    gpu.host_register %3 : memref<*xf32>
    %4 = memref.cast %1 : memref<32x32xf32> to memref<*xf32>
    gpu.host_register %4 : memref<*xf32>
    %5 = memref.cast %2 : memref<32x32xf32> to memref<*xf32>
    gpu.host_register %5 : memref<*xf32>
    %6 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
    %7 = memref.cast %6 : memref<1xf32> to memref<*xf32>
    gpu.host_register %7 : memref<*xf32>
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
      memref.store %cst, %0[%arg0, %arg1] : memref<32x32xf32>
      scf.yield
    }
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
      memref.store %cst, %1[%arg0, %arg1] : memref<32x32xf32>
      scf.yield
    }
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
      memref.store %cst_0, %2[%arg0, %arg1] : memref<32x32xf32>
      scf.yield
    }
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        memref.store %cst_0, %6[%c0] {to = "sum_rv"} : memref<1xf32>
        affine.for %arg2 = 0 to 32 {
          %9 = memref.load %0[%arg0, %arg2] {from = "A"} : memref<32x32xf32>
          %10 = memref.load %1[%arg2, %arg1] {from = "B"} : memref<32x32xf32>
          %11 = arith.mulf %9, %10 : f32
          %12 = memref.load %6[%c0] {from = "sum_rv"} : memref<1xf32>
          %13 = arith.addf %11, %12 : f32
          memref.store %13, %6[%c0] {to = "sum_rv"} : memref<1xf32>
        } {loop_name = "k", reduction}
        %8 = memref.load %6[%c0] {from = "sum_rv"} : memref<1xf32>
        memref.store %8, %2[%arg0, %arg1] {to = "C"} : memref<32x32xf32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "C"}
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
}

