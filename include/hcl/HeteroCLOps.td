//===- HeteroCLOps.td - hcl dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_OPS
#define HCL_OPS

include "HeteroCLDialect.td"
include "HeteroCLTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def HeteroCL_FooOp : HeteroCL_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `hcl.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = hcl.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def HeteroCL_LoopHandleOp : HeteroCL_Op<"loop_handle"> 
{
    let summary = "loop handle";
    let description = [{
        loop handle
    }];

    let arguments = (ins );
    let results = (outs LoopType:$result);
    let assemblyFormat = [{
       attr-dict `:` type($result)
    }];
}

def HeteroCL_ReorderOp : HeteroCL_Op<"reorder">
{
    let summary = "placeholder";
    let description = [{
        reorder
    }];

    let arguments = (ins Confined<ArrayAttr, [ArrayMinCount<2>]>:$loop_id);
    let results = (outs );
    let assemblyFormat = [{
        attr-dict
    }];
}

def HeteroCL_SplitOp : HeteroCL_Op<"split"> 
{
    let summary = "split";
    let description = [{
        split
    }];

    let arguments = (ins LoopType:$loop, UI32Attr:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `:` type($loop) `,` $factor `)` attr-dict
    }];
    // https://mlir.llvm.org/docs/OpDefinitions/#custom-builder-methods
    // let builders = [];
}

def HeteroCL_TileOp : HeteroCL_Op<"tile"> 
{
    let summary = "tile";
    let description = [{
        tile
    }];

    let arguments = (ins LoopType:$x_loop, LoopType:$y_loop, UI32Attr:$x_factor, UI32Attr:$y_factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $x_loop `:` type($x_loop) `,` $y_loop `:` type($y_loop) `,` $x_factor `,` $y_factor `)` attr-dict
    }];
}

#endif // HCL_OPS
