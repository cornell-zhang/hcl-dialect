module {
    func @matrix_multiply(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        // %l4 = hcl.fuse (%l1, %l2: !hcl.LoopHandle<"i">, !hcl.LoopHandle<"j">) -> !hcl.LoopHandle<"i_j_fused">
        // (i,j)->(ij/1024,ij%1024)
        %l4 = hcl.fuse (%l1, %l2, %l3: !hcl.LoopHandle<"i">, !hcl.LoopHandle<"j">, !hcl.LoopHandle<"k">) -> !hcl.LoopHandle<"i_j_k_fused">
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return %C : memref<1024x1024xf32>
    }
}