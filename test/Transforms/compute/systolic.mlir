// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func.func @conv1d(%A: memref<64xf32>, %W: memref<3xf32>, %C: memref<61xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"

        // Loop nest with attr from PoCC/isl
        affine.for %i = 0 to 61 {
            affine.for %j = 0 to 3 {
                %a = affine.load %A[%i+%j] : memref<64xf32>
                %b = affine.load %W[%j] : memref<3xf32>
                %c = affine.load %C[%i] : memref<61xf32>
                %prod = arith.mulf %a, %b : f32
                %sum = arith.addf %prod, %c: f32
                affine.store %sum, %C[%i] : memref<61xf32>
            // CHECK:  } {dep_disstance = 1 : i64, loop_name = "j", unroll = 3 : i32}
            } { loop_name = "j", dep_distance = 1 }
        } { loop_name = "i", op_name = "s", dep_distance = 0 }

        %pe_array = hcl.unfold( %lj, 3 ) 
        hcl.to(%W : memref<3xf32>, %pe_array) { pe_index = [0,1,2] } -> memref<1xf32>
        %pe0_w = hcl.to(%W: memref<3xf32>, %pe_array) { pe_index = [0] } -> memref<1xf32>
        %pe1_w = hcl.to(%pe0_w: memref<1xf32>, %pe_array) { pe_index = [1] } -> memref<1xf32>
        return
    }
}