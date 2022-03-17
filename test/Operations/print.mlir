// RUN: hcl-opt %s 

module {

  memref.global "private" @gv0 : memref<4x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>

  func @top() -> () {
    %0 = memref.get_global @gv0 : memref<4x4xf32>
    hcl.print(%0) : memref<4x4xf32>
    return
  }
}
