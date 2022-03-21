// RUN: hcl-opt -jit %s | FileCheck %s

module {

  memref.global "private" @gv0 : memref<4x4xf64> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>

  func @top() -> () {
    %0 = memref.get_global @gv0 : memref<4x4xf64>
// CHECK: 1.000000 2.000000 3.000000 4.000000
// CHECK: 1.000000 2.000000 3.000000 4.000000
// CHECK: 1.000000 2.000000 3.000000 4.000000
// CHECK: 1.000000 2.000000 3.000000 4.000000
    hcl.print(%0) : memref<4x4xf64>
    return
  }
}
