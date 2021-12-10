//===- HeteroCLOps.td - hcl dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "HeteroCLDialect.td"
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

def HeteroCL_GetLoop : HeteroCL_Op<"getLoop", [NoSideEffect, SameOperandsAndResultType]> {
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

def HeteroCL_CreateHandle : HeteroCL_Op<"create_handle"> 
{
    let summary = "placeholder";
    let description = [{
        blank
    }];

    let arguments = (ins );
    let results = (outs AnyType:$out1);

    // let builders = [OpBuilder<
    //     "Builder builder, OperationState &result, int64_t loop, int64_t block_size"
    // >];

    let assemblyFormat = [{
       attr-dict `:` type($out1)
    }];
}


def HeteroCL_LoopReorder : HeteroCL_Op<"reorder">
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

def HeteroCL_LoopSplit : HeteroCL_Op<"split"> 
{
    let summary = "placeholder";
    let description = [{
        split
    }];

    let arguments = (ins I32:$loop, I32:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `:` type($loop) `,` $factor `:` type($factor) `)` attr-dict
    }];
}

// def HandleType : HeteroCL_Type<"handle"> {
//     let mnemonic = "handle";

//     let summary = "An integer type with special semantics";

//     let description = [{
//         An alternate integer type. This type differentiates itself from the
//         standard integer type by not having a SignednessSemantics parameter, just
//         a width.
//     }];

//     let parameters = (ins "unsigned":$width);

//     // We define the printer inline.
//     let printer = [{
//         $_printer << "int<" << getImpl()->width << ">";
//     }];

//     // The parser is defined here also.
//     let parser = [{
//         if ($_parser.parseLess())
//         return Type();
//         int width;
//         if ($_parser.parseInteger(width))
//         return Type();
//         if ($_parser.parseGreater())
//         return Type();
//         return get($_ctxt, width);
//     }];
// }



#endif // STANDALONE_OPS
