// RUN: hcl-opt %s --lower-to-llvm
module {
  func @top(%arg0: memref<10xi32>) -> memref<10xi32> attributes {bit, extra_itypes = "s", extra_otypes = "s", top} {
    %0 = memref.alloc() {name = "compute_1"} : memref<10xi32>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xi32>
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      %c0 = arith.constant 0 : index
      %3 = hcl.get_bit(%2 : i32, %c0) -> i1
      %4 = arith.extsi %3 : i1 to i32
      affine.store %4, %0[%arg1] {to = "compute_1"} : memref<10xi32>
    } {loop_name = "x", stage_name = "compute_1"}
    return %0 : memref<10xi32>
  }
}