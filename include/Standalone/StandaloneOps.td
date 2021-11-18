//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "StandaloneDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



def Standalone_FooOp : Standalone_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Standalone_GetLoop : Standalone_Op<"getLoop", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "placeholder";
    let description = [{
        Get loop handle from loop id
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Standalone_LoopSplit : Standalone_Op<"split"> 
{
    let summary = "placeholder";
    let description = [{
        split
    }];

    let arguments = (ins AnyType:$loop, I32:$block_size);
    let results = (outs AnyType:$out1);

    // let builders = [OpBuilder<
    //     "Builder builder, OperationState &result, int64_t loop, int64_t block_size"
    // >];

    let assemblyFormat = [{
        $loop $block_size attr-dict `:` type($loop) type($block_size) type($out1)
    }];
}

def Standalone_LoopReorder : Standalone_Op<"reorder"> 
{
    let summary = "placeholder";
    let description = [{
        reorder
    }];

    let arguments = (ins AnyType:$in1, AnyType:$in2, AnyType:$in3, AnyType:$in4);
    let results = (outs );

    let assemblyFormat = [{
        `[` $in1 `,` $in2 `,` $in3 `,` $in4 `]` attr-dict `:` type($in1) type($in2) type($in3) type($in4)
    }];
}


#endif // STANDALONE_OPS
