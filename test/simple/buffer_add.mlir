// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @add_buffer_at_axis_0(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                // B[i, j] = A[i, j] + 1
                %a = affine.load %A[%i, %j] : memref<1024x1024xf32>
                %cst = constant 1.0 : f32
                %sum = addf %a, %cst: f32 //register
                affine.store %sum, %B[%i, %j] : memref<1024x1024xf32>
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.buffer_at(%B: memref<1024x1024xf32>, 0)
        return %C : memref<1024x1024xf32>
    }
    // expected:
    // func @add_buffer_at_axis_0(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    // {
    //     %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
    //     %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
    //     affine.for %i = 0 to 1024 {
    //         %buf_B = memref.alloc() : memref<1024xf32>
    //         affine.for %j = 0 to 1024 {
    //             affine.store 0, %buf_B[%j] : memref<1024xf32>
    //         } { loop_name = "j_init" }
    //         affine.for %j = 0 to 1024 {
    //             // B[i, j] = A[i, j] + 1
    //             %a = affine.load %A[%i, %j] : memref<1024x1024xf32>
    //             %sum = addf %a, 1: f32 //register
    //             affine.store %sum, %buf_B[%j] : memref<1024x1024xf32>
    //         } { loop_name = "j" }
    //         affine.for %j = 0 to 1024 {
    //             affine.store %buf_B[%j], %B[%i, %j] : memref<1024x1024xf32>
    //         } { loop_name = "j_back" }
    //     } { loop_name = "i" }
    //     return %C : memref<1024x1024xf32>
    // }
    func @add_buffer_at_axis_1(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                // B[i, j] = A[i, j] + 1
                %a = affine.load %A[%i, %j] : memref<1024x1024xf32>
                %cst = constant 1.0 : f32
                %sum = addf %a, %cst: f32 //register
                affine.store %sum, %B[%i, %j] : memref<1024x1024xf32>
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.buffer_at(%B: memref<1024x1024xf32>, 1)
        return %C : memref<1024x1024xf32>
    }
}