// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @matrix_multiply( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>
    {
        %l1 = hcl.loop_handle {} : !hcl.Loop<"i">
        %l2 = hcl.loop_handle {} : !hcl.Loop<"j">
        %l3 = hcl.loop_handle {} : !hcl.Loop<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32>
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.split (%l3: !hcl.Loop<"k">, 16)
        return %C : tensor<?x?xf32>
    }
}