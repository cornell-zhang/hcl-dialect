module {
    func @matrix_multiply( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>
    {
        %l1 = hcl.create_loop_handle {} : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle {} : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle {} : !hcl.LoopHandle<"k">
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
        // %l4 = hcl.fuse (%l1, %l2: !hcl.LoopHandle<"i">, !hcl.LoopHandle<"j">) -> !hcl.LoopHandle<"i_j_fused">
        // (i,j)->(ij/1024,ij%1024)
        %l4 = hcl.fuse (%l1, %l2, %l3: !hcl.LoopHandle<"i">, !hcl.LoopHandle<"j">, !hcl.LoopHandle<"k">) -> !hcl.LoopHandle<"i_j_k_fused">
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return %C : tensor<?x?xf32>
    }
}