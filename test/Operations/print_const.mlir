// RUN: hcl-opt -jit %s | FileCheck %s

// This file tests hcl.print operations.
// - Different types: int, float, fixed
// - Different print formats
// - Different number of values to print

module {
    memref.global "private" @gv_cst : memref<1xi64> = dense<[8]>
    func.func @top () -> () {
        %fixed_memref = hcl.get_global_fixed @gv_cst : memref<1x!hcl.Fixed<32,2>>

        %c1_i32 = arith.constant 144 : i32
        hcl.print(%c1_i32) : i32
        return
    }

}