// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @matrix_multiply(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                // B[i, j] = A[i, j] + 1
                %a = affine.load %A[%i, %j] : memref<?x?xf32>
                %cst = constant 1.0 : f32
                %sum = addf %a, %cst: f32 //register
                affine.store %sum, %B[%i, %j] : memref<?x?xf32>
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.buffer_at(%B: memref<?x?xf32>, 0)
        return %C : memref<?x?xf32>
    }
    // generated
    // func @matrix_multiply(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32>
    // {
    //     %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
    //     %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
    //     affine.for %i = 0 to 1024 {
    //         %buf_B = memref.alloc() : memref<?xf32>
    //         affine.for %j = 0 to 1024 {
    //             affine.store 0, %buf_B[%j] : memref<?xf32>
    //         } { loop_name = "j_init" }
    //         affine.for %j = 0 to 1024 {
    //             // B[i, j] = A[i, j] + 1
    //             %a = affine.load %A[%i, %j] : memref<?x?xf32>
    //             %sum = addf %a, 1: f32 //register
    //             affine.store %sum, %buf_B[%j] : memref<?x?xf32>
    //         } { loop_name = "j" }
    //         affine.for %j = 0 to 1024 {
    //             affine.store %buf_B[%j], %B[%i, %j] : memref<?x?xf32>
    //         } { loop_name = "j_back" }
    //     } { loop_name = "i" }
    //     return %C : memref<?x?xf32>
    // }
}