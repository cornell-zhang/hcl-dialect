// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @gemm_fuse_two(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        // CHECK: affine.for %[[ARG:.*]] = 0 to 1048576 {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s, %li, %lj)
        // (i,j)->(ij/1024,ij%1024)
        return
    }
    func @gemm_fuse_three(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        // CHECK: affine.for %[[ARG:.*]] = 0 to 536870912 {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s, %li, %lj, %lk)
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return
    }
    func @gemm_fuse_two_among_four(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %ll = hcl.create_loop_handle "l" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        // CHECK: affine.for %[[ARG:.*]] = 0 to 1024 {
        affine.for %i = 0 to 1024 {
            // CHECK: affine.for %[[ARG1:.*]] = 0 to 1048576 {
            affine.for %j = 0 to 1024 {
                affine.for %l = 0 to 1024 {
                    affine.for %k = 0 to 512 {
                        %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                        %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                        %prod = mulf %a, %b : f32
                        %sum = addf %prod, %c: f32
                        affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                    } { loop_name = "k" }
                } { loop_name = "l" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s, %lj, %ll)
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return
    }
}