// RUN: hcl-opt --fixed-to-integer --jit %s | FileCheck %s
module {
  memref.global "private" @gv_cst : memref<2x2xi64> = dense<[[8, 0], [10, 20]]>

  func.func @top() -> () {
    %0 = hcl.get_global_fixed @gv_cst : memref<2x2x!hcl.Fixed<32,2>>
    hcl.print (%0) {format = "%.1f \n"} : memref<2x2x!hcl.Fixed<32,2>> 
    return
  }
}

// CHECK: 2.0
// CHECK: 0.0
// CHECK: 2.5
// CHECK: 5.0