// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func @vector_add(%A: memref<256xf32>, %B: memref<256xf32>, %C: memref<256xf32>)
    {
        %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        // CHECK: affine.for %arg3 = 0 to 4 {
        // CHECK:   affine.for %arg4 = 0 to 64 {
        affine.for %i = 0 to 256 {
            %a = affine.load %A[%i] : memref<256xf32>
            %b = affine.load %B[%i] : memref<256xf32>
            %sum = arith.addf %a, %b : f32
            affine.store %sum, %C[%i] : memref<256xf32>
            // CHECK:     } {loop_name = "i.inner", thread_axis = 3 : i32}
            // CHECK:   } {loop_name = "i.outer", stage_name = "s", thread_axis = 0 : i32}
        } { loop_name = "i", stage_name = "s" }

        %li_outer, %li_inner = hcl.split (%s, %li, 64)
        hcl.bind (%s, %li_outer, "BlockIdxX")
        hcl.bind (%s, %li_inner, "ThreadIdxX")
        return
    }
}
