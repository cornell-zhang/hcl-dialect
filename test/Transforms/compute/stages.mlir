// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @matrix_multiply(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %l1 = hcl.create_loop_handle "i"
        %l2 = hcl.create_loop_handle "j"
        %l3 = hcl.create_loop_handle "k"
        %s1 = hcl.create_op_handle "s1"
        %l11 = hcl.create_loop_handle "i1"
        %l21 = hcl.create_loop_handle "j1"
        %l31 = hcl.create_loop_handle "k1"
        %s2 = hcl.create_op_handle "s2"
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK: } {loop_name = "j"}
                } { loop_name = "k" }
            // CHECK: } {loop_name = "k"}
            } { loop_name = "j" }
        // CHECK: } {loop_name = "i", op_name = "s1"}
        } { loop_name = "i", op_name = "s1" }
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK: } {loop_name = "i1"}
                } { loop_name = "k1" }
            // CHECK: } {loop_name = "j1"}
            } { loop_name = "j1" }
        // CHECK: } {loop_name = "k1", op_name = "s2"}
        } { loop_name = "i1", op_name = "s2"}
        hcl.reorder (%s1, %l3, %l2)
        hcl.reorder (%s2, %l31, %l21, %l11)
        return
    }
}