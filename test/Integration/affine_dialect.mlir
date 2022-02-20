module {
  func @top(%arg0: memref<16x22xf32>, %arg1: memref<22x18xf32>, %arg2: memref<18x24xf32>, %arg3: memref<16x24xf32>) -> memref<16x24xf32> attributes {llvm.emit_c_interface, top} {
    %0 = memref.alloc() {name = "D"} : memref<16x24xf32>
    %1 = memref.alloc() {name = "C"} : memref<18x24xf32>
    %2 = memref.alloc() {name = "A"} : memref<16x22xf32>
    %3 = memref.alloc() {name = "B"} : memref<22x18xf32>
    %4 = memref.alloc() {name = "out_AB"} : memref<16x18xf32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 18 {
        %7 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %7[%c0] {to = "sum_rv"} : memref<1xf32>
        affine.for %arg6 = 0 to 22 {
          %9 = affine.load %2[%arg4, %arg6] {from = "A"} : memref<16x22xf32>
          %10 = affine.load %3[%arg6, %arg5] {from = "B"} : memref<22x18xf32>
          %11 = arith.mulf %9, %10 : f32
          %12 = affine.load %7[%c0] {from = "sum_rv"} : memref<1xf32>
          %13 = arith.addf %11, %12 : f32
          affine.store %13, %7[%c0] {to = "sum_rv"} : memref<1xf32>
        } {loop_name = "r"}
        %c0_0 = arith.constant 0 : index
        %8 = affine.load %7[%c0_0] {from = "sum_rv"} : memref<1xf32>
        affine.store %8, %4[%arg4, %arg5] {to = "out_AB"} : memref<16x18xf32>
      } {loop_name = "y"}
    } {loop_name = "x", stage_name = "out_AB"}
    %5 = memref.alloc() {name = "out_ABC"} : memref<16x24xf32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 24 {
        %7 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %7[%c0] {to = "sum_rv"} : memref<1xf32>
        affine.for %arg6 = 0 to 18 {
          %9 = affine.load %4[%arg4, %arg6] {from = "out_AB"} : memref<16x18xf32>
          %10 = affine.load %1[%arg6, %arg5] {from = "C"} : memref<18x24xf32>
          %11 = arith.mulf %9, %10 : f32
          %12 = affine.load %7[%c0] {from = "sum_rv"} : memref<1xf32>
          %13 = arith.addf %11, %12 : f32
          affine.store %13, %7[%c0] {to = "sum_rv"} : memref<1xf32>
        } {loop_name = "k"}
        %c0_0 = arith.constant 0 : index
        %8 = affine.load %7[%c0_0] {from = "sum_rv"} : memref<1xf32>
        affine.store %8, %5[%arg4, %arg5] {to = "out_ABC"} : memref<16x24xf32>
      } {loop_name = "y"}
    } {loop_name = "x", stage_name = "out_ABC"}
    %6 = memref.alloc() {name = "E"} : memref<16x24xf32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 24 {
        %cst = arith.constant 1.000000e-01 : f32
        %7 = affine.load %5[%arg4, %arg5] {from = "out_ABC"} : memref<16x24xf32>
        %8 = arith.mulf %cst, %7 : f32
        %cst_0 = arith.constant 1.000000e-01 : f32
        %9 = affine.load %0[%arg4, %arg5] {from = "D"} : memref<16x24xf32>
        %10 = arith.mulf %cst_0, %9 : f32
        %11 = arith.addf %8, %10 : f32
        affine.store %11, %6[%arg4, %arg5] {to = "E"} : memref<16x24xf32>
      } {loop_name = "y"}
    } {loop_name = "x", stage_name = "E"}

    %u_memref = memref.cast %6 : memref<16x24xf32> to memref<*xf32>
    call @print_memref_f32(%u_memref) : (memref<*xf32>) -> ()
    return %6 : memref<16x24xf32>
  }
  func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
}