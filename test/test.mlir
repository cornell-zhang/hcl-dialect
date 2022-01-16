module  {
  func @top() {
    %0 = memref.alloc() : memref<32x32xf32>
    %1 = memref.alloc() : memref<32x32xf32>
    %2 = hcl.create_stage_handle "B" : !hcl.StageHandle
    %3 = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %4 = hcl.create_loop_handle "j" : !hcl.LoopHandle
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %9 = affine.load %0[%arg0, %arg1] : memref<32x32xf32>
        %cst = constant 1.000000e+00 : f32
        %10 = addf %9, %cst : f32
        affine.store %10, %1[%arg0, %arg1] : memref<32x32xf32>
      } {loop_name = "j"}
    } {loop_name = "i", stage_name = "B"}
    %5 = memref.alloc() : memref<32x32xf32>
    %6 = hcl.create_stage_handle "C" : !hcl.StageHandle
    %7 = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %8 = hcl.create_loop_handle "j" : !hcl.LoopHandle
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %9 = affine.load %0[%arg0, %arg1] : memref<32x32xf32>
        %cst = constant 1.000000e+00 : f32
        %10 = addf %9, %cst : f32
        affine.store %10, %5[%arg0, %arg1] : memref<32x32xf32>
      } {loop_name = "j"}
    } {loop_name = "i", stage_name = "C"}
    hcl.compute_at(%2, %6, %8)
    return
  }
}