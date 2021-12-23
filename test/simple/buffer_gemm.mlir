// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @gemm_buffer_at_axis_1(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.buffer_at(%C: memref<1024x1024xf32>, 1)
        return %C : memref<1024x1024xf32>
    }
    // expected:
    // func @gemm_buffer_at_axis_1(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    // {
    //     %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
    //     %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
    //     %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
    //     affine.for %i = 0 to 1024 {
    //         affine.for %j = 0 to 1024 {
    //             %buf_C = memref.alloc() : memref<1xf32>
    //             %zero = constant 0.0 : f32
    //             %idx = constant 0 : index
    //             affine.store %zero, %buf_C[%idx] : memref<1xf32>
    //             affine.for %k = 0 to 1024 {
    //                 %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
    //                 %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
    //                 %c = affine.load %buf_C[%idx] : memref<1xf32>
    //                 %prod = mulf %a, %b : f32
    //                 %sum = addf %prod, %c: f32
    //                 affine.store %sum, %buf_C[%idx] : memref<1xf32>
    //             } { loop_name = "k" }
    //             %buf_C_fin = affine.load %buf_C[%idx] : memref<1xf32>
    //             affine.store %buf_C_fin, %C[%i, %j] : memref<1024x1024xf32>
    //         } { loop_name = "j" }
    //     } { loop_name = "i" }
    //     return %C : memref<1024x1024xf32>
    // }
    func @gemm_buffer_at_axis_0(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.buffer_at(%C: memref<1024x1024xf32>, 0)
        return %C : memref<1024x1024xf32>
    }
    // storing at reduction axis is prohibited
    func @gemm_buffer_at_axis_2(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.buffer_at(%C: memref<1024x1024xf32>, 2)
        return %C : memref<1024x1024xf32>
    }
}