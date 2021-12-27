module {
    func @matrix_multiply(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %s1 = hcl.create_stage_handle { stage_name = "s1" }: !hcl.StageHandle
        %s2 = hcl.create_stage_handle { stage_name = "s2" }: !hcl.StageHandle
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
        } { loop_name = "i", stage_name = "s1" }
        %li = hcl.create_loop_handle : !hcl.LoopHandle<"i1">
        %lj = hcl.create_loop_handle : !hcl.LoopHandle<"j1">
        %lk = hcl.create_loop_handle : !hcl.LoopHandle<"k1">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k1" }
            } { loop_name = "j1" }
        } { loop_name = "i1", stage_name = "s2" }
        %li_outer, %li_inner = hcl.split (%s2: !hcl.StageHandle, %li: !hcl.LoopHandle<"i1">, 8) -> (!hcl.LoopHandle<"i1.outer">, !hcl.LoopHandle<"i1.inner">)
        hcl.reorder (%s2: !hcl.StageHandle, %lk, %lj, %li_inner: !hcl.LoopHandle<"k1">, !hcl.LoopHandle<"j1">, !hcl.LoopHandle<"i1.inner">)
        return %C : memref<1024x1024xf32>
    }
}