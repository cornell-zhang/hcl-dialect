// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                     %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                     %arg4: memref<10x10xf32>) {
        %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %l2 = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %l3 = hcl.create_loop_handle "i1" : !hcl.LoopHandle
        %l4 = hcl.create_loop_handle "k1" : !hcl.LoopHandle
        %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
        %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
        // CHECK: affine.for %arg5 = 0 to 3 {
        // CHECK:   affine.for %arg6 = 0 to 3 {
        affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
            %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
            %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
            %2 = arith.mulf %0, %1 : f32
            affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
            } {loop_name = "k"}
        } {loop_name = "i", stage_name = "s1"}
        affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
            %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
            %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
            } {loop_name = "k1"}
        } {loop_name = "i1", stage_name = "s2"}
        hcl.compute_at (%s1, %s2, %l4)
        return
    }
    func @matrix_multiply( %A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>, %D: memref<1024x1024xf32>)
    {
        %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %l2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %l3 = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %l4 = hcl.create_loop_handle "i1" : !hcl.LoopHandle
        %l5 = hcl.create_loop_handle "j1" : !hcl.LoopHandle
        %l6 = hcl.create_loop_handle "k1" : !hcl.LoopHandle
        %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
        %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
        // C=A*B
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s1" }
        // D=C*A
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %c = affine.load %C[%k, %j] : memref<1024x1024xf32>
                    %d = affine.load %D[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %c : f32
                    %sum = arith.addf %prod, %d: f32
                    affine.store %sum, %D[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k1" }
            } { loop_name = "j1" }
        } { loop_name = "i1", stage_name = "s2" }
        hcl.compute_at (%s1, %s2, %l6)
        return
    }
}