// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @matrix_multiply(%A: tensor<1024x512xf32>, %B: tensor<512x1024xf32>, %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
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
        %li_outer, %li_inner = hcl.split (%s, %li, 8)
        %li_in_out, %li_in_in = hcl.split (%s, %li_inner, 16) // nest with split
        %li_out_out, %li_out_in = hcl.split (%s, %li_outer, 2) // multiple split
        %lj_out, %lj_in, %lk_out, %lk_in = hcl.tile (%s, %lj, %lk, 2, 4) // split & tile
        // %l14, %l15, %l16, %l17 = hcl.tile (%li_in_out, %li_in_in, 2, 2) // nest with split (failed)
        hcl.unroll (%s, %lk_in, 16) // unroll
        hcl.pipeline (%s, %lk_out, 1) // pipeline
        hcl.parallel (%s, %lj_in) // parallel
        return %C : tensor<1024x1024xf32>
    }
}