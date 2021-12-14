// RUN: hcl-opt %s | hcl-opt | FileCheck %s

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
        %l4, %l5 = hcl.split (%l1: !hcl.LoopHandle<"i">, 8) -> (!hcl.LoopHandle<"i.outer">, !hcl.LoopHandle<"i.inner">)
        %l6, %l7 = hcl.split (%l5: !hcl.LoopHandle<"i.inner">, 16) -> (!hcl.LoopHandle<"i.inner.outer">, !hcl.LoopHandle<"i.inner.inner">) // nest with split
        %l8, %l9 = hcl.split (%l4: !hcl.LoopHandle<"i.outer">, 2) -> (!hcl.LoopHandle<"i.outer.outer">, !hcl.LoopHandle<"i.outer.inner">) // multiple split
        %l10, %l11, %l12, %l13 = hcl.tile (%l2: !hcl.LoopHandle<"j">, %l3: !hcl.LoopHandle<"k">, 2, 4) -> (!hcl.LoopHandle<"j.outer">, !hcl.LoopHandle<"j.inner">, !hcl.LoopHandle<"k.outer">, !hcl.LoopHandle<"k.inner">) // split & tile
        // %l14, %l15, %l16, %l17 = hcl.tile (%l6: !hcl.LoopHandle<"i.inner.outer">, %l7: !hcl.LoopHandle<"i.inner.inner">, 2, 2) -> (!hcl.LoopHandle<"i.inner.outer.outer">, !hcl.LoopHandle<"i.inner.outer.inner">, !hcl.LoopHandle<"i.inner.inner.outer">, !hcl.LoopHandle<"i.inner.outer">) // nest with split (failed)
        hcl.unroll (%l13: !hcl.LoopHandle<"k.inner">, 16) // unroll
        hcl.pipeline (%l12: !hcl.LoopHandle<"k.outer">, 1) // pipeline
        hcl.parallel (%l11: !hcl.LoopHandle<"j.inner">) // parallel
        return %C : tensor<?x?xf32>
    }
}