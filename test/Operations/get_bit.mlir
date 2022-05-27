// RUN: hcl-opt %s --jit | FileCheck %s
module {
  memref.global "private" @gv0 : memref<1xi32> = dense<[3]>
  func @top() -> () attributes {bit, itypes = "s", otypes = "s", top} {
    %0 = memref.get_global @gv0 : memref<1xi32>
    %res =memref.alloc() : memref<1xi32>
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %0[%arg1] : memref<1xi32>
      %c1_i32 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %3 = hcl.get_bit(%1 : i32, %c0) -> i1
      %4 = arith.extui %3 : i1 to i32
      affine.store %4, %res[%arg1] : memref<1xi32>
    } 
// CHECK: 1
    hcl.print(%res) {format="%.0f \n"}: memref<1xi32>
    return
  }
}