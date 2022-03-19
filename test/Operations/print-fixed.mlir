// RUN: hcl-opt -jit %s | FileCheck %s

module {
  func @top(%0: memref<4x4x!hcl.Fixed<4, 2>>) -> () {
    hcl.print(%0) : memref<4x4x!hcl.Fixed<4, 2>>
    return
  }
}
