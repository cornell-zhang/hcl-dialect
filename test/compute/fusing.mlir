module {
    func @gemm_fuse_two(%A: tensor<1024x512xf32>, %B: tensor<512x1024xf32>, %C: tensor<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x512xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<512x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    tensor.insert %sum into %C[%i, %j] : tensor<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s, %li, %lj)
        // (i,j)->(ij/1024,ij%1024)
        return
    }
    func @gemm_fuse_three(%A: tensor<1024x512xf32>, %B: tensor<512x1024xf32>, %C: tensor<1024x1024xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %lk = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x512xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<512x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    tensor.insert %sum into %C[%i, %j] : tensor<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s, %li, %lj, %lk)
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return
    }
}