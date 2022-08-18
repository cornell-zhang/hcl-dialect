// RUN: hcl-opt %s --fixed-to-integer --jit | FileCheck %s
module {
  memref.global "private" @fixed_gv : memref<2x2xi64> = dense<[[8, 0], [10, 20]]>
  func.func @top() -> () {
    %0 = hcl.get_global_fixed @fixed_gv : memref<2x2x!hcl.Fixed<32,2>>
    %1 = memref.alloc() : memref<2x2xi32>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %3 = affine.load %0[%arg0, %arg1] : memref<2x2x!hcl.Fixed<32,2>>
        %4 = hcl.fixed_to_int (%3) : !hcl.Fixed<32, 2> -> i32
        affine.store %4, %1[%arg0, %arg1] : memref<2x2xi32>
      }
    }
    hcl.print(%1) {format = "%.2f \n"} : memref<2x2xi32> 
    return
  }
}

// CHECK: 2.00
// CHECK: 0.00
// CHECK: 2.00
// CHECK: 5.00