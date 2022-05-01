// RUN: hcl-opt --lower-composite --lower-to-llvm %s
module {
  // func @top (%arg0 : !hcl.struct<!hcl.struct<i3, memref<10xf32>>, i3>) -> () {
  //   %0 = hcl.struct_get %arg0[0] : !hcl.struct<!hcl.struct<i3, memref<10xf32>>, i3> -> !hcl.struct<i3, memref<10xf32>>
  //   %zero = arith.constant 0 : i32
  //   %1 = hcl.struct_construct (%0, %zero) : !hcl.struct<i3, memref<10xf32>>, i32 -> !hcl.struct<!hcl.struct<i3, memref<10xf32>>, i32>
  //   // %one = arith.constant 1 : i32
  //   // %2 = hcl.struct_set %arg0[1], %one : !hcl.struct<!hcl.struct<i3, memref<10xf32>>, i3>, i32 -> !hcl.struct<!hcl.struct<i3, memref<10xf32>>, i32>
  //   return
  // }

  func @top () -> () {
    %1 = arith.constant 0 : i32
    %2 = arith.constant 1 : i32
    %3 = hcl.struct_construct(%1, %2) : i32, i32 -> !hcl.struct<i32, i32>
    %4 = hcl.struct_get %3[0] : !hcl.struct<i32, i32> -> i32
    %5 = hcl.struct_get %3[1] : !hcl.struct<i32, i32> -> i32
    %6 = arith.addi %4, %5 : i32
    return
  }

  func @nested_struct() -> () {
    %1 = arith.constant 0 : i32
    %2 = arith.constant 1 : i32
    %3 = hcl.struct_construct(%1, %2) : i32, i32 -> !hcl.struct<i32, i32>
    %4 = hcl.struct_construct(%3, %2) : !hcl.struct<i32, i32>, i32 -> !hcl.struct<!hcl.struct<i32, i32>, i32>
    %5 = hcl.struct_get %4[0] : !hcl.struct<!hcl.struct<i32, i32>, i32> -> !hcl.struct<i32, i32>
    %6 = hcl.struct_get %5[0] : !hcl.struct<i32, i32> -> i32
    %7 = arith.addi %6, %2 : i32
    return
  }
} 
