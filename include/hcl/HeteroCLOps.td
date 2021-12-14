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

def HeteroCL_CreateLoopHandleOp : HeteroCL_Op<"create_loop_handle"> 
{
    let summary = "loop handle";
    let description = [{
        loop handle
    }];

    let arguments = (ins );
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
       attr-dict `:` type($result)
    }];
}

def HeteroCL_ReorderOp : HeteroCL_Op<"reorder">
{
    let summary = "reorder";
    let description = [{
        reorder
    }];

    // variadic
    let arguments = (ins Variadic<LoopHandle>:$loops);
    let results = (outs );
    // optional groups
    // https://mlir.llvm.org/docs/OpDefinitions/#optional-groups
    // TODO: should at least have one input
    let assemblyFormat = [{
        `(` ($loops^ `:` type($loops))? `)` attr-dict
    }];
}

def HeteroCL_SplitOp : HeteroCL_Op<"split"> 
{
    let summary = "split";
    let description = [{
        split
    }];

    let arguments = (ins LoopHandle:$loop, UI32Attr:$factor);
    let results = (outs LoopHandle:$outer, LoopHandle:$inner);
    let assemblyFormat = [{
        `(` $loop `:` type($loop) `,` $factor `)` attr-dict `->` `(` type($outer) `,` type($inner) `)`
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

    let arguments = (ins LoopHandle:$x_loop, LoopHandle:$y_loop, UI32Attr:$x_factor, UI32Attr:$y_factor);
    let results = (outs LoopHandle:$x_outer, LoopHandle:$x_inner, LoopHandle:$y_outer, LoopHandle:$y_inner);
    let assemblyFormat = [{
        `(` $x_loop `:` type($x_loop) `,` $y_loop `:` type($y_loop) `,` $x_factor `,` $y_factor `)` attr-dict `->` `(` type($x_outer) `,` type($x_inner) `,` type($y_outer) `,` type($y_inner) `)`
    }];
}

def HeteroCL_FuseOp : HeteroCL_Op<"fuse"> 
{
    let summary = "fuse";
    let description = [{
        fuse
    }];

    let arguments = (ins Variadic<LoopHandle>:$loops);
    let results = (outs LoopHandle:$fused_loop);
    let assemblyFormat = [{
        `(` ($loops^ `:` type($loops))? `)` attr-dict `->` type($fused_loop)
    }];
}

def HeteroCL_ComputeAtOp : HeteroCL_Op<"compute_at"> 
{
    let summary = "compute_at";
    let description = [{
        compute_at
    }];

    let arguments = (ins LoopHandle:$loop1, LoopHandle:$loop2);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop1 `:` type($loop1) `,` $loop2 `:` type($loop2) `)` attr-dict
    }];
}

def HeteroCL_UnrollOp : HeteroCL_Op<"unroll"> 
{
    let summary = "unroll";
    let description = [{
        unroll
    }];

    // factor=0 means fully unroll the loop
    let arguments = (ins LoopHandle:$loop, DefaultValuedAttr<UI32Attr,"0">:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `:` type($loop) `,` $factor `)` attr-dict
    }];
}

def HeteroCL_PipelineOp : HeteroCL_Op<"pipeline"> 
{
    let summary = "pipeline";
    let description = [{
        pipeline
    }];

    let arguments = (ins LoopHandle:$loop, DefaultValuedAttr<UI32Attr, "1">:$ii);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `:` type($loop) `,` $ii `)` attr-dict
    }];
    let builders = [
        OpBuilder<(ins "hcl::LoopHandleType":$loop,
                   CArg<"uint64_t", "1">:$ii)>
    ];
}

def HeteroCL_ParallelOp : HeteroCL_Op<"parallel"> 
{
    let summary = "parallel";
    let description = [{
        parallel
    }];

    let arguments = (ins LoopHandle:$loop);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `:` type($loop) `)` attr-dict
    }];
}

#endif // HCL_OPS
