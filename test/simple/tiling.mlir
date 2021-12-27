// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    func @matrix_multiply(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
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
        %li_outer, %li_inner = hcl.split (%s: !hcl.StageHandle, %li: !hcl.LoopHandle<"i">, 8) -> (!hcl.LoopHandle<"i.outer">, !hcl.LoopHandle<"i.inner">)
        %li_in_out, %li_in_in = hcl.split (%s: !hcl.StageHandle, %li_inner: !hcl.LoopHandle<"i.inner">, 16) -> (!hcl.LoopHandle<"i.inner.outer">, !hcl.LoopHandle<"i.inner.inner">) // nest with split
        %li_out_out, %li_out_in = hcl.split (%s: !hcl.StageHandle, %li_outer: !hcl.LoopHandle<"i.outer">, 2) -> (!hcl.LoopHandle<"i.outer.outer">, !hcl.LoopHandle<"i.outer.inner">) // multiple split
        %lj_out, %lj_in, %lk_out, %lk_in = hcl.tile (%s: !hcl.StageHandle, %lj: !hcl.LoopHandle<"j">, %lk: !hcl.LoopHandle<"k">, 2, 4) -> (!hcl.LoopHandle<"j.outer">, !hcl.LoopHandle<"j.inner">, !hcl.LoopHandle<"k.outer">, !hcl.LoopHandle<"k.inner">) // split & tile
        // %l14, %l15, %l16, %l17 = hcl.tile (%li_in_out: !hcl.LoopHandle<"i.inner.outer">, %li_in_in: !hcl.LoopHandle<"i.inner.inner">, 2, 2) -> (!hcl.LoopHandle<"i.inner.outer.outer">, !hcl.LoopHandle<"i.inner.outer.inner">, !hcl.LoopHandle<"i.inner.inner.outer">, !hcl.LoopHandle<"i.inner.outer">) // nest with split (failed)
        hcl.unroll (%lk_in: !hcl.LoopHandle<"k.inner">, 16) // unroll
        hcl.pipeline (%lk_out: !hcl.LoopHandle<"k.outer">, 1) // pipeline
        hcl.parallel (%lj_in: !hcl.LoopHandle<"j.inner">) // parallel
        return %C : memref<1024x1024xf32>
    }
}