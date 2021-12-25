// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @gemm_interleaving_accu(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k", reduction = 1 : i32}
            } { loop_name = "j" }
        } { loop_name = "i" }
        hcl.reorder(%l3, %l2: !hcl.LoopHandle<"k">, !hcl.LoopHandle<"j">)
        hcl.buffer_at(%C: memref<1024x1024xf32>, 0)
        hcl.pipeline(%l2: !hcl.LoopHandle<"j">, 1)
        return %C : memref<1024x1024xf32>
    }
}