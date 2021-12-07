// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @matrix_multiply( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) 
	-> tensor<?x?xf32>
    { 
        %li = hcl.create_handle : i32
        %lj = hcl.create_handle : i32
        %lk = hcl.create_handle : i32
        
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32> 
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32> 
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32> 
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } 
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32> 
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32> 
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32> 
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } 
            } 
        } {loop_handle =  "l1"} // l1.i, l1.j, l1.k.0 l1.k.1

        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32> 
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32> 
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32> 
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } {loop_handle =  2 }
            } {loop_handle =  1 } 
        } {loop_handle =  0 }
        %block_size = constant 8 : i32
        %ii0 = hcl.split %li %block_size : i32 i32 i32
        %ii1 = hcl.split %lj %block_size : i32 i32 i32
        hcl.reorder [%li, %lj, %ii0, %ii1] : i32 i32 i32 i32
        return %C : tensor<?x?xf32>
    }

}
