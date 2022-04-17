// RUN: hcl-opt %s --jit | FileCheck %s
// Input: 0x0000
// By setting the third bit to 1, we get
// Output: 0x0004
module {
  memref.global "private" @gv0 : memref<1xi32> = dense<[0]>
  func @top() -> () attributes {bit, extra_itypes = "s", extra_otypes = "s", top} {
    %0 = memref.get_global @gv0 : memref<1xi32>
    %res =memref.alloc() : memref<1xi32>
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %0[%arg1] : memref<1xi32>
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %val = arith.constant 1 : i1
      hcl.set_bit(%1 : i32, %c2, %val : i1)
      affine.store %1, %res[%arg1] : memref<1xi32>
    } 
// CHECK: 4
    hcl.print(%res) {format="%.0f \n"}: memref<1xi32>
    return
  }
}