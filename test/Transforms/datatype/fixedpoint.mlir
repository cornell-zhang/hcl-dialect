// RUN: hcl-opt %s --fixed-to-integer
module {

  func.func @issue_56(%arg0: memref<1000x!hcl.Fixed<8, 6>>) -> memref<1000x!hcl.Fixed<8, 6>> attributes {itypes = "_", otypes = "_", llvm.emit_c_interface, top} {
    %0 = memref.alloc() {name = "compute_1"} : memref<1000x!hcl.Fixed<8, 6>>
    affine.for %arg1 = 0 to 1000 {
      %2 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<1000x!hcl.Fixed<8, 6>>
      affine.store %2, %0[%arg1] {to = "compute_1"} : memref<1000x!hcl.Fixed<8, 6>>
    } {loop_name = "x", op_name = "compute_1"}
    %1 = memref.alloc() {name = "compute_2"} : memref<1000x!hcl.Fixed<8, 6>>
    affine.for %arg1 = 0 to 1000 {
      %2 = affine.load %0[%arg1] {from = "compute_1"} : memref<1000x!hcl.Fixed<8, 6>>
      affine.store %2, %1[%arg1] {to = "compute_2"} : memref<1000x!hcl.Fixed<8, 6>>
    } {loop_name = "x", op_name = "compute_2"}
    return %1 : memref<1000x!hcl.Fixed<8, 6>>
  }

  func.func @func_call(%arg0: memref<10xi32>, %arg1: memref<10xi32>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 10 {
        func.call @Stage_update_B(%arg0, %arg1, %arg3) {inputs = "compute_0,compute_1,"} : (memref<10xi32>, memref<10xi32>, index) -> ()
      } {loop_name = "loop_1"}
    } {loop_name = "loop_0"}
    return
  }
  func.func @Stage_update_B(%arg0: memref<10xi32>, %arg1: memref<10xi32>, %arg2: index) attributes {itypes = "sss"} {
    %0 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10xi32>
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.addi %0, %c1_i32 : i32
    affine.store %1, %arg1[%arg2] {to = "compute_1"} : memref<10xi32>
    return
  }


  func.func @no_return(%arg0: memref<10x!hcl.Fixed<32, 2>>, %arg1: memref<10x!hcl.Fixed<32, 2>>, %arg3: memref<10x!hcl.Fixed<32, 2>>) -> () {
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!hcl.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!hcl.Fixed<32, 2>>
      %3 = "hcl.add_fixed"(%1, %2) : (!hcl.Fixed<32, 2>, !hcl.Fixed<32, 2>) -> !hcl.Fixed<32, 2>
      affine.store %3, %arg3[%arg2] {to = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return
  }

  func.func @top_vadd(%arg0: memref<10x!hcl.Fixed<32, 2>>, %arg1: memref<10x!hcl.Fixed<32, 2>>) -> memref<10x!hcl.Fixed<32, 2>> {
    %0 = memref.alloc() {name = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!hcl.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!hcl.Fixed<32, 2>>
      %3 = "hcl.add_fixed"(%1, %2) : (!hcl.Fixed<32, 2>, !hcl.Fixed<32, 2>) -> !hcl.Fixed<32, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<10x!hcl.Fixed<32, 2>>
  }


  func.func @top_vmul(%arg0: memref<10x!hcl.Fixed<32, 2>>, %arg1: memref<10x!hcl.Fixed<32, 2>>) -> memref<10x!hcl.Fixed<32, 2>> {
    %0 = memref.alloc() {name = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!hcl.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!hcl.Fixed<32, 2>>
      %3 = "hcl.mul_fixed"(%1, %2) : (!hcl.Fixed<32, 2>, !hcl.Fixed<32, 2>) -> !hcl.Fixed<32, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10x!hcl.Fixed<32, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<10x!hcl.Fixed<32, 2>>
  }

  func.func @no_change_int(%arg0: memref<10xi32>) -> memref<10xi32> attributes {itypes = "s", otypes = "s"} {
    %0 = memref.alloc() {name = "compute_1"} : memref<10xi32>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xi32>
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      affine.store %2, %0[%arg1] {to = "compute_1"} : memref<10xi32>
    } {loop_name = "x", op_name = "compute_1"}
    return %0 : memref<10xi32>
  }
  func.func @no_change_float(%arg0: memref<10xf32>) -> memref<10xf32> attributes {itypes = "_", otypes = "_"} {
    %0 = memref.alloc() {name = "compute_1"} : memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xf32>
      %cst = arith.constant 5.000000e-01 : f32
      %2 = arith.cmpf ogt, %1, %cst : f32
      %3 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %4 = arith.select %2, %3, %cst_0 : f32
      affine.store %4, %0[%arg1] {to = "compute_1"} : memref<10xf32>
    } {loop_name = "x", op_name = "compute_1"}
    return %0 : memref<10xf32>
  }

  func.func @fixed_div(%arg0: memref<100x!hcl.Fixed<6, 2>>, %arg1: memref<100x!hcl.Fixed<6, 2>>) -> memref<100x!hcl.Fixed<6, 2>> attributes {itypes = "__", otypes = "_"} {
    %0 = memref.alloc() {name = "compute_2"} : memref<100x!hcl.Fixed<6, 2>>
    affine.for %arg2 = 0 to 100 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<100x!hcl.Fixed<6, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<100x!hcl.Fixed<6, 2>>
      %3 = "hcl.div_fixed"(%1, %2) : (!hcl.Fixed<6, 2>, !hcl.Fixed<6, 2>) -> !hcl.Fixed<6, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<100x!hcl.Fixed<6, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<100x!hcl.Fixed<6, 2>>
  }
}