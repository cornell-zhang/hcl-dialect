// RUN: hcl-opt -opt %s | FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1) -> (d0 mod 4, d1 mod 4, d0 floordiv 4, d1 floordiv 4)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0 floordiv 256, d1 floordiv 512, d0 mod 256, d1 floordiv 512)>
// CHECK: #map2 = affine_map<(d0, d1) -> (d0, 0, 0, d1)>
module {
    func @matrix_multiply(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
        %l2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
        %l3 = hcl.create_loop_handle "k" : !hcl.LoopHandle
        %s = hcl.create_stage_handle "s" : !hcl.StageHandle
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        hcl.partition(%A: memref<1024x1024xf32>, "CyclicPartition", 0, 4)
        hcl.partition(%B: memref<1024x1024xf32>, "BlockPartition", 2, 2)
        // expected-error@+1 {{Partition on the array partitioned before.The original layout map will be rewritten!}}
        hcl.partition(%B: memref<1024x1024xf32>, "BlockPartition", 1, 4)
        hcl.partition(%C: memref<1024x1024xf32>, "CompletePartition", 1)
        // CHECK: return %arg2 : memref<1024x1024xf32, #map2>
        return %C : memref<1024x1024xf32>
    }
}