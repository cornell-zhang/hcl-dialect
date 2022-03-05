// RUN: hcl-opt %s
module {
  // func @top_vadd(%arg0: memref<10x!hcl.Fixed<32, 2>>, %arg1: memref<10x!hcl.Fixed<32, 2>>) -> memref<10x!hcl.Fixed<32, 2>> {
  //   %0 = memref.alloc() {name = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
  //   affine.for %arg2 = 0 to 10 {
  //     %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!hcl.Fixed<32, 2>>
  //     %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!hcl.Fixed<32, 2>>
  //     %3 = "hcl.add_fixed"(%1, %2) : (!hcl.Fixed<32, 2>, !hcl.Fixed<32, 2>) -> !hcl.Fixed<32, 2>
  //     affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
  //   } {loop_name = "x", stage_name = "compute_2"}
  //   return %0 : memref<10x!hcl.Fixed<32, 2>>
  // }

  // func @top_vmul(%arg0: memref<10x!hcl.Fixed<32, 2>>, %arg1: memref<10x!hcl.Fixed<32, 2>>) -> () {
  func @top_vmul(%arg0: memref<10x!hcl.Fixed<32, 2>>, %arg1: memref<10x!hcl.Fixed<32, 2>>) -> memref<10x!hcl.Fixed<32, 2>> {
    %0 = memref.alloc() {name = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!hcl.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!hcl.Fixed<32, 2>>
      %3 = "hcl.mul_fixed"(%1, %2) : (!hcl.Fixed<32, 2>, !hcl.Fixed<32, 2>) -> !hcl.Fixed<32, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    } {loop_name = "x", stage_name = "compute_2"}
    return %0 : memref<10x!hcl.Fixed<32, 2>>
    // return
  }
}