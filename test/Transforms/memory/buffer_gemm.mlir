// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @gemm_buffer_at_axis_0(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            // CHECK: {{.*}} = memref.alloc() : memref<1024xf32>
            // CHECK: %cst = arith.constant 0.000000e+00 : f32
            // CHECK: affine.for %[[VAR:.*]] = 0 to 1024 {
            // CHECK:     affine.store %cst, {{.*}}[%[[VAR]]] : memref<1024xf32>
            // CHECK: } {loop_name = "j_init", pipeline_ii = 1 : i32}
            // CHECK: affine.for %[[VAR]] = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k", reduction = 1 : i32}
            } { loop_name = "j" }
            // CHECK: } {loop_name = "j"}
            // CHECK: affine.for %[[VAR]] = 0 to 1024 {
            // CHECK:     %[[RES:.*]] = affine.load {{.*}}[%[[VAR]]] : memref<1024xf32>
            // CHECK:     affine.store %[[RES]], {{.*}}[{{.*}}, {{.*}}] : memref<1024x1024xf32>
            // CHECK: } {loop_name = "j_back", pipeline_ii = 1 : i32}
        } { loop_name = "i", stage_name = "s" }
        %buf = hcl.buffer_at(%s, %C: memref<1024x1024xf32>, %li) -> memref<1024xf32>
        return
    }
    func @gemm_buffer_at_axis_1(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                // CHECK: %[[VAR:.*]] = memref.alloc() : memref<1xf32>
                // CHECK: %cst = arith.constant 0.000000e+00 : f32
                // CHECK: %c0 = arith.constant 0 : index
                // CHECK: affine.store %cst, %[[VAR]][%c0] : memref<1xf32>
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k", reduction = 1 : i32}
                // CHECK: } {loop_name = "k", reduction = 1 : i32}
                // CHECK: %[[RES:.*]] = affine.load %[[VAR]][%c0] : memref<1xf32>
                // CHECK: affine.store %[[RES]], {{.*}}[{{.*}}, {{.*}}] : memref<1024x1024xf32>
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %buf = hcl.buffer_at(%s, %C: memref<1024x1024xf32>, %lj) -> memref<1xf32>
        return
    }
    // Notice: storing at reduction axis is prohibited
    // func @gemm_buffer_at_axis_2(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    // {
    //     %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    //     %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    //     %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
    //     %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    //     affine.for %i = 0 to 1024 {
    //         affine.for %j = 0 to 1024 {
    //             affine.for %k = 0 to 512 {
    //                 %a = affine.load %A[%i, %k] : memref<1024x512xf32>
    //                 %b = affine.load %B[%k, %j] : memref<512x1024xf32>
    //                 %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
    //                 %prod = arith.mulf %a, %b : f32
    //                 %sum = arith.addf %prod, %c: f32
    //                 affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
    //             } { loop_name = "k", reduction = 1 : i32}
    //         } { loop_name = "j" }
    //     } { loop_name = "i", stage_name = "s" }
    //     %buf = hcl.buffer_at(%s, %C: memref<1024x1024xf32>, 2) -> memref<1xf32>
    //     return
    // }
    func @gemm_interleaving_accu(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            // CHECK: {{.*}} = memref.alloc() : memref<1024xf32>
            // CHECK: %cst = arith.constant 0.000000e+00 : f32
            // CHECK: affine.for %[[VAR:.*]] = 0 to 1024 {
            // CHECK: } {loop_name = "j_init", pipeline_ii = 1 : i32}
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k", reduction = 1 : i32}
            } { loop_name = "j" }
            // CHECK:     } {loop_name = "j", pipeline_ii = 1 : i32}
            // CHECK: } {loop_name = "k", reduction = 1 : i32}
            // CHECK: affine.for %[[VAR]] = 0 to 1024 {
            // CHECK: } {loop_name = "j_back", pipeline_ii = 1 : i32}
        } { loop_name = "i", stage_name = "s" }
        hcl.reorder(%s, %lk, %lj)
        %buf = hcl.buffer_at(%s, %C: memref<1024x1024xf32>, %li) -> memref<1024xf32>
        hcl.pipeline(%s, %lj, 1)
        return
    }
    // func @tiled_gemm_interleaving_accu(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    // {
    //     %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    //     %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    //     %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
    //     %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    //     affine.for %i = 0 to 1024 {
    //         affine.for %j = 0 to 1024 {
    //             affine.for %k = 0 to 512 {
    //                 %a = affine.load %A[%i, %k] : memref<1024x512xf32>
    //                 %b = affine.load %B[%k, %j] : memref<512x1024xf32>
    //                 %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
    //                 %prod = arith.mulf %a, %b : f32
    //                 %sum = arith.addf %prod, %c: f32
    //                 affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
    //             } { loop_name = "k", reduction = 1 : i32}
    //         } { loop_name = "j" }
    //     } { loop_name = "i", stage_name = "s" }
    //     %lj_out, %lj_in = hcl.split(%s, %lj, 2)
    //     hcl.reorder(%s, %lk, %lj_in)
    //     %buf = hcl.buffer_at(%s, %C: memref<1024x1024xf32>, %lj) -> memref<1024xf32>
    //     hcl.pipeline(%s, %lj_in, 1)
    //     return %C : memref<1024x1024xf32>
    // }
}