// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @blur(%A: memref<10x10xf32>, %B: memref<10x8xf32>) -> memref<10x8xf32>
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 10 {
            affine.for %j = 0 to 8 {
                %tmp = affine.load %A[%i, %j] : memref<10x10xf32>
                %tmp1 = affine.load %A[%i, %j+1] : memref<10x10xf32>
                %tmp2 = affine.load %A[%i, %j+2] : memref<10x10xf32>
                %sum = addf %tmp, %tmp1: f32
                %sum1 = addf %sum, %tmp2: f32
                affine.store %sum1, %B[%i, %j] : memref<10x8xf32>
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %buf = hcl.reuse_at(%s, %A: memref<10x10xf32>, 1) -> memref<10x8xf32>
        return %B : memref<10x8xf32>
    }
}