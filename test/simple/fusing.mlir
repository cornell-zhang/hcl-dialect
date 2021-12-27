module {
    func @gemm_fuse_two(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %li = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %lj = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %lk = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        %s = hcl.create_stage_handle { stage_name = "s" }: !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s: !hcl.StageHandle, %li, %lj: !hcl.LoopHandle<"i">, !hcl.LoopHandle<"j">) -> !hcl.LoopHandle<"i_j_fused">
        // (i,j)->(ij/1024,ij%1024)
        return %C : memref<1024x1024xf32>
    }
    func @gemm_fuse_three(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %li = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %lj = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %lk = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        %s = hcl.create_stage_handle { stage_name = "s" }: !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        %l_fused = hcl.fuse (%s: !hcl.StageHandle, %li, %lj, %lk: !hcl.LoopHandle<"i">, !hcl.LoopHandle<"j">, !hcl.LoopHandle<"k">) -> !hcl.LoopHandle<"i_j_k_fused">
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return %C : memref<1024x1024xf32>
    }
}