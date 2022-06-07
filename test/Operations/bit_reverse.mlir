// RUN: hcl-opt %s
module {
  func @top(%arg0: memref<10xi8>, %arg1: memref<10xi8>) attributes {bit, itypes = "uu", otypes = ""} {
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %arg0[%arg2] {from = "compute_0", unsigned} : memref<10xi8>
      %1 = hcl.bit_reverse(%0 : i8) {unsigned}
      %2 = affine.load %arg1[%arg2] {from = "compute_1", unsigned} : memref<10xi8>
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      hcl.set_slice(%2 : i8, %c7, %c0, %1 : i8)
      affine.store %2, %arg1[%arg2] {to = "compute_1", unsigned} : memref<10xi8>
    } {loop_name = "loop_0"}
    return
  }
}