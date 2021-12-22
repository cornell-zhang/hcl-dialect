module {
    func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                     %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                     %arg4: memref<10x10xf32>) {
        %l1 = hcl.create_loop_handle {} : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle {} : !hcl.LoopHandle<"k">
        %l3 = hcl.create_loop_handle {} : !hcl.LoopHandle<"i1">
        %l4 = hcl.create_loop_handle {} : !hcl.LoopHandle<"k1">
        affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
            %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
            %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
            %2 = mulf %0, %1 : f32
            affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
            } {loop_name = "k"}
        } {loop_name = "i"}
        affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
            %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
            %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
            %2 = addf %0, %1 : f32
            affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
            } {loop_name = "k1"}
        } {loop_name = "i1"}
        hcl.compute_at (%l1: !hcl.LoopHandle<"i">, %l3: !hcl.LoopHandle<"i1">)
        return
    }
    func @matrix_multiply( %A: tensor<1024x1024xf32>, %B: tensor<1024x1024xf32>, %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle {} : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle {} : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle {} : !hcl.LoopHandle<"k">
        %l4 = hcl.create_loop_handle {} : !hcl.LoopHandle<"i1">
        %l5 = hcl.create_loop_handle {} : !hcl.LoopHandle<"j1">
        %l6 = hcl.create_loop_handle {} : !hcl.LoopHandle<"k1">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x1024xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<1024x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x1024xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<1024x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } { loop_name = "k1" }
            } { loop_name = "j1" }
        } { loop_name = "i1" }
        hcl.compute_at (%l1: !hcl.LoopHandle<"i">, %l4: !hcl.LoopHandle<"i1">)
        return %C : tensor<1024x1024xf32>
    }
}