// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @matrix_multiply(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %l2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %l3 = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
        %l11 = hcl.create_loop_handle "i1" : !hcl.LoopHandle
        %l21 = hcl.create_loop_handle "j1" : !hcl.LoopHandle
        %l31 = hcl.create_loop_handle "k1" : !hcl.LoopHandle
        %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK: } {loop_name = "j"}
                } { loop_name = "k" }
            // CHECK: } {loop_name = "k"}
            } { loop_name = "j" }
        // CHECK: } {loop_name = "i", stage_name = "s1"}
        } { loop_name = "i", stage_name = "s1" }
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK: } {loop_name = "i1"}
                } { loop_name = "k1" }
            // CHECK: } {loop_name = "j1"}
            } { loop_name = "j1" }
        // CHECK: } {loop_name = "k1", stage_name = "s2"}
        } { loop_name = "i1", stage_name = "s2"}
        hcl.reorder (%s1, %l3, %l2)
        hcl.reorder (%s2, %l31, %l21, %l11)
        return
    }
}