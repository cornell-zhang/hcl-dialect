// RUN: hcl-opt %s --affine-to-gpu

// This is an example of gemm on GPU.
// The affine memrory accesses are manually converted to memref load/store.
// The reduction loop's result memref is manually promoted to a scalar.

module {
  func @top(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) -> memref<32x32xf32> attributes {itypes = "__", otypes = "_"} {
    %0 = memref.alloc() {name = "C"} : memref<32x32xf32>
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        %1 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        memref.store %cst, %1[%c0] {to = "sum_rv"} : memref<1xf32>
        affine.for %arg4 = 0 to 32 {
          %3 = memref.load %arg0[%arg2, %arg4] {from = "A"} : memref<32x32xf32>
          %4 = memref.load %arg1[%arg4, %arg3] {from = "B"} : memref<32x32xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = memref.load %1[%c0] {from = "sum_rv"} : memref<1xf32>
          %7 = arith.addf %5, %6 : f32
          memref.store %7, %1[%c0] {to = "sum_rv"} : memref<1xf32>
        } {loop_name = "k", reduction}
        %2 = memref.load %1[%c0] {from = "sum_rv"} : memref<1xf32>
        memref.store %2, %0[%arg2, %arg3] {to = "C"} : memref<32x32xf32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "C"}
    return %0 : memref<32x32xf32>
  }
}