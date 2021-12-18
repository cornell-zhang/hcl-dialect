// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @matrix_multiply(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<?x?xf32>
                    %b = affine.load %B[%k, %j] : memref<?x?xf32>
                    %c = affine.load %C[%i, %j] : memref<?x?xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<?x?xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i" }
        %l4, %l5 = hcl.split (%l1: !hcl.LoopHandle<"i">, 8) -> (!hcl.LoopHandle<"i.outer">, !hcl.LoopHandle<"i.inner">)
        %l6, %l7, %l8, %l9 = hcl.tile (%l2: !hcl.LoopHandle<"j">, %l3: !hcl.LoopHandle<"k">, 2, 4) -> (!hcl.LoopHandle<"j.outer">, !hcl.LoopHandle<"j.inner">, !hcl.LoopHandle<"k.outer">, !hcl.LoopHandle<"k.inner">) // split & tile
        %l10, %l11 = hcl.split (%l6: !hcl.LoopHandle<"j.outer">, 16) -> (!hcl.LoopHandle<"j.outer.outer">, !hcl.LoopHandle<"j.outer.inner">) 
        return %C : memref<?x?xf32>
    }
}