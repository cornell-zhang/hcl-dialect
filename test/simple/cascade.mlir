// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @matrix_multiply( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>
    {
        %l1 = hcl.create_loop_handle : !hcl.LoopHandle<"i">
        %l2 = hcl.create_loop_handle : !hcl.LoopHandle<"j">
        %l3 = hcl.create_loop_handle : !hcl.LoopHandle<"k">
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = tensor.extract %A[%i, %k] : tensor<?x?xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<?x?xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<?x?xf32>
                    %prod = mulf %a, %b : f32
                    %sum  = addf %prod, %c: f32
                } { loop_name = "k" } // #hcl.Loop<"k"> } //: hcl.LoopHandle<"k"> }
            } { loop_name = "j" }
        } { loop_name = "i" }
        %l4, %l5 = hcl.split (%l1: !hcl.LoopHandle<"i">, 8) -> (!hcl.LoopHandle<"i.outer">, !hcl.LoopHandle<"i.inner">)
        %l6, %l7, %l8, %l9 = hcl.tile (%l2: !hcl.LoopHandle<"j">, %l3: !hcl.LoopHandle<"k">, 2, 4) -> (!hcl.LoopHandle<"j.outer">, !hcl.LoopHandle<"j.inner">, !hcl.LoopHandle<"k.outer">, !hcl.LoopHandle<"k.inner">) // split & tile
        %l10, %l11 = hcl.split (%l6: !hcl.LoopHandle<"j.outer">, 16) -> (!hcl.LoopHandle<"j.outer.outer">, !hcl.LoopHandle<"j.outer.inner">) 
        return %C : tensor<?x?xf32>
    }
}