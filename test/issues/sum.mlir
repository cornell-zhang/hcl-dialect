// RUN: hcl-opt -jit %s | FileCheck %s

// CHECK: module {
module  {
memref.global "private" @gv0 : memref<2xf32> = dense<[1.0, 2.0]>
  func @top(%arg0: memref<2xf32>) -> memref<1xf32> {
    //%0 = hcl.create_stage_handle "sum" : !hcl.StageHandle
    %1 = memref.alloc() : memref<1xf32>
    //%2 = hcl.create_loop_handle "_" : !hcl.LoopHandle
    affine.for %arg1 = 0 to 1 {
      %3 = memref.alloc() : memref<1xf32>
      %c3_0 = constant 0 : index
      %c3 = constant 0.0 : f32
      affine.store %c3, %3[%c3_0] : memref<1xf32>
      %U3 = memref.cast %3 : memref<1xf32> to memref<*xf32>
      call @print_memref_f32(%U3) : (memref<*xf32>) -> ()
      affine.for %arg2 = 0 to 2 {
        %5 = affine.load %arg0[%arg2] : memref<2xf32>
        %c0_0 = constant 0 : index
        %6 = affine.load %3[%c0_0] : memref<1xf32>
        %7 = addf %5, %6 : f32
        affine.store %7, %3[%c0_0] : memref<1xf32>
      } {loop_name = "x"}
      %c0 = constant 0 : index
      %4 = affine.load %3[%c0] : memref<1xf32>
      affine.store %4, %1[%arg1] : memref<1xf32>
    } {loop_name = "_", stage_name = "sum"}
    return %1 : memref<1xf32>
  }

  func @main() {
    %0 = memref.get_global @gv0 : memref<2xf32>
    %1 = call @top(%0) : (memref<2xf32>) -> (memref<1xf32>)
    %U = memref.cast %1 : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%U) : (memref<*xf32>) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
}

