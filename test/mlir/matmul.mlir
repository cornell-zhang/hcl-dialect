// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @matrix_multiply( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) 
	-> tensor<?x?xf32>
    { 
        %l0 = constant 0 : i32
        %l1 = constant 1 : i32
        %l2 = constant 2 : i32
        %li, %lj, %lk = affine.for %i = 0 to 1024 iter_args(%label_i0=%l0, %label_i1=%l1, %label_i2=%l2) -> (i32, i32, i32) {
            %lj:2 = affine.for %j = 0 to 1024 iter_args(%label_j0=%l1, %label_j1=%l2) -> (i32, i32)  {
                %lk = affine.for %k = 0 to 1024 iter_args(%label_k0=%l2) -> i32 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32> 
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32> 
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32> 
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                    //%0 = tensor.insert %sum into %C[%i, %j] : tensor<?x?xf32> 
                    affine.yield %label_k0 : i32
                }
                affine.yield %label_j0, %label_j1 : i32, i32
            } 
            affine.yield %label_i0, %label_i1, %label_i2 : i32, i32, i32
        }
        %block_size = constant 8 : i32
        %ii0 = hcl.split %li %block_size : i32 i32 i32
        %ii1 = hcl.split %lj %block_size : i32 i32 i32
        hcl.reorder [%li, %lj, %ii0, %ii1] : i32 i32 i32 i32
        return %C : tensor<?x?xf32>
    }

}
