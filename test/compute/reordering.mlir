// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @gemm(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK:   } {loop_name = "i.inner"}
                // CHECK: } {loop_name = "j"}
                } { loop_name = "k" }
            // CHECK: } {loop_name = "k"}
            } { loop_name = "j" }
        // CHECK: } {loop_name = "i.outer", stage_name = "s"}
        } { loop_name = "i", stage_name = "s" }
        %li_outer, %li_inner = hcl.split (%s, %li, 8)
        hcl.reorder (%s, %lk, %lj, %li_inner)
        return
    }
    func @gemm_reorder_outermost(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK: } {loop_name = "j"}
                } { loop_name = "k" }
            // CHECK: } {loop_name = "i"}
            } { loop_name = "j" }
        // CHECK: } {loop_name = "k", stage_name = "s"}
        } { loop_name = "i", stage_name = "s" }
        hcl.reorder (%s, %lk, %li, %lj)
        return
    }
}