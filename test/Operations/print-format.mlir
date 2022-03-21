// RUN: hcl-opt -jit %s | FileCheck %s

module {

  memref.global "private" @gv0 : memref<4x4xf64> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>

  func @top() -> () {
    %0 = memref.get_global @gv0 : memref<4x4xf64>
// CHECK: 1.0
// CHECK: 2.0
// CHECK: 3.0
// CHECK: 4.0
    hcl.print(%0) {format = "%.1f \n"} : memref<4x4xf64> 
    return
  }
}
