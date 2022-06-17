// RUN: hcl-opt -opt %s | FileCheck %s

module {
  func @top(%arg0: memref<10x32xi32>) -> memref<10x32xi32> attributes {itypes = "s", otypes = "s"} {
    %0 = memref.alloc() {name = "C"} : memref<10x32xi32>
    %1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %arg0[%arg1, %arg2] {from = "A"} : memref<10x32xi32>
        %c1_i32 = arith.constant 1 : i32
        %13 = arith.addi %12, %c1_i32 : i32
        affine.store %13, %0[%arg1, %arg2] {to = "C"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", stage_name = "C"}
    %3 = hcl.create_stage_handle "C" : !hcl.StageHandle
    %4 = memref.alloc() {name = "D"} : memref<10x32xi32>
    %5 = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %6 = hcl.create_loop_handle "j" : !hcl.LoopHandle
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %0[%arg1, %arg2] {from = "C"} : memref<10x32xi32>
        %c2_i32 = arith.constant 2 : i32
        %13 = arith.muli %12, %c2_i32 : i32
        affine.store %13, %4[%arg1, %arg2] {to = "D"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", stage_name = "D"}
    %7 = hcl.create_stage_handle "D" : !hcl.StageHandle
    %8 = memref.alloc() {name = "E"} : memref<10x32xi32>
    %9 = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %10 = hcl.create_loop_handle "j" : !hcl.LoopHandle
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %4[%arg1, %arg2] {from = "D"} : memref<10x32xi32>
        %c3_i32 = arith.constant 3 : i32
        %13 = arith.muli %12, %c3_i32 : i32
        affine.store %13, %8[%arg1, %arg2] {to = "E"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", stage_name = "E"}
    %11 = hcl.create_stage_handle "E" : !hcl.StageHandle
    // CHECK: call @Stage_C
    hcl.outline (%3)
    // CHECK: call @Stage_D_E
    hcl.outline (%7, %11)
    return %8 : memref<10x32xi32>
  }
}