// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @test(%A: tensor<1024x512x!hcl.Fixed<12,6>>, %B: tensor<512x1024x!hcl.UFixed<12,2>>)
    {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                %a = tensor.extract %A[%i, %j] : tensor<1024x512x!hcl.Fixed<12,6>>
                %b = tensor.extract %B[%i, %j] : tensor<512x1024x!hcl.UFixed<12,2>>
            }
        }
        return
    }
}