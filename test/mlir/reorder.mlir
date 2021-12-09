// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @matrix_multiply( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) 
	-> tensor<?x?xf32>
    { 
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32> 
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32> 
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32> 
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                    %0 = tensor.insert %sum into %C[%i, %j] : tensor<?x?xf32> 
                } {loop_handle =  "k"}
            } {loop_handle =  "j"} 
        } {loop_handle =  "i"}

        hcl.reorder {loop_id=["i", "k", "j"]}
        return %C : tensor<?x?xf32>
    }

}
