// RUN: hcl-opt %s --fixed-to-integer --jit | FileCheck %s
module {
  memref.global "private" @int_gv : memref<2x2xi32> = dense<[[2, 4], [5, 15]]>
  func.func @top() -> () {
    %0 = memref.get_global @int_gv : memref<2x2xi32>
    %1 = memref.alloc() : memref<2x2x!hcl.Fixed<32, 2>>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %3 = affine.load %0[%arg0, %arg1] : memref<2x2xi32>
        %4 = hcl.int_to_fixed (%3) : i32 -> !hcl.Fixed<32, 2>
        affine.store %4, %1[%arg0, %arg1] : memref<2x2x!hcl.Fixed<32, 2>>
      }
    }
    hcl.print(%1) {format = "%.0f \n"} : memref<2x2x!hcl.Fixed<32,2>> 
    return
  }
}

// CHECK: 2
// CHECK: 4
// CHECK: 5
// CHECK: 15