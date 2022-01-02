module {
    func @matrix_multiply(%A: tensor<1024x1024xf32>, %B: tensor<1024x1024xf32>, %C: tensor<1024x1024xf32>)
    {
        %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %l2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %l3 = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
        %l11 = hcl.create_loop_handle "i1" : !hcl.LoopHandle
        %l21 = hcl.create_loop_handle "j1" : !hcl.LoopHandle
        %l31 = hcl.create_loop_handle "k1" : !hcl.LoopHandle
        %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x1024xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<1024x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    tensor.insert %sum into %C[%i, %j] : tensor<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s1" }
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x1024xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<1024x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    tensor.insert %sum into %C[%i, %j] : tensor<1024x1024xf32>
                } { loop_name = "k1" }
            } { loop_name = "j1" }
        } { loop_name = "i1", stage_name = "s2"}
        hcl.reorder (%s1, %l3, %l2)
        hcl.reorder (%s2, %l31, %l21, %l11)
        return
    }
}