// RUN: hcl-opt %s --jit | FileCheck %s
// Input: 0x0000
// By setting the 3,2,1 bits to 110, we get
// Output: 12 (0x000C)
module {
  memref.global "private" @gv0 : memref<1xi32> = dense<[0]>
  func.func @top() -> () attributes {bit, itypes = "s", otypes = "s", top} {
    %0 = memref.get_global @gv0 : memref<1xi32>
    %res =memref.alloc() : memref<1xi32>
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %0[%arg1] : memref<1xi32>
      %lo = arith.constant 1 : index
      %hi = arith.constant 3 : index
      %val = arith.constant 6 : i3
      hcl.set_slice(%1 : i32, %hi, %lo, %val : i3)
      affine.store %1, %res[%arg1] : memref<1xi32>
    } 
// CHECK: 12
    hcl.print(%res) {format="%.0f \n"}: memref<1xi32>
    return
  }
}