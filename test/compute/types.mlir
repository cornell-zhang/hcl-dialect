// RUN: hcl-opt -opt %s | FileCheck %s

module {
    // CHECK: func @test(%arg0: tensor<1024x512x!hcl.fixed<12, 6>>, %arg1: tensor<512x1024x!hcl.ufixed<12, 2>>) {
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