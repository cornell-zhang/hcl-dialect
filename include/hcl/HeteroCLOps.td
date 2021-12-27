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
include "HeteroCLAttrs.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

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

def HeteroCL_CreateStageHandleOp : HeteroCL_Op<"create_stage_handle"> 
{
    let summary = "stage handle";
    let description = [{
        stage handle
    }];

    let arguments = (ins );
    let results = (outs StageHandle:$result);
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
    let arguments = (ins StageHandle:$stage, Variadic<LoopHandle>:$loops);
    let results = (outs );
    // optional groups
    // https://mlir.llvm.org/docs/OpDefinitions/#optional-groups
    // TODO: should at least have one input
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` ($loops^ `:` type($loops))? `)` attr-dict
    }];
}

def HeteroCL_SplitOp : HeteroCL_Op<"split"> 
{
    let summary = "split";
    let description = [{
        split
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, UI32Attr:$factor);
    let results = (outs LoopHandle:$outer, LoopHandle:$inner);
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` $loop `:` type($loop) `,` $factor `)` attr-dict `->` `(` type($outer) `,` type($inner) `)`
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

    let arguments = (ins StageHandle:$stage, LoopHandle:$x_loop, LoopHandle:$y_loop, UI32Attr:$x_factor, UI32Attr:$y_factor);
    let results = (outs LoopHandle:$x_outer, LoopHandle:$x_inner, LoopHandle:$y_outer, LoopHandle:$y_inner);
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` $x_loop `:` type($x_loop) `,` $y_loop `:` type($y_loop) `,` $x_factor `,` $y_factor `)` attr-dict `->` `(` type($x_outer) `,` type($x_inner) `,` type($y_outer) `,` type($y_inner) `)`
    }];
}

def HeteroCL_FuseOp : HeteroCL_Op<"fuse"> 
{
    let summary = "fuse";
    let description = [{
        fuse
    }];

    let arguments = (ins StageHandle:$stage, Variadic<LoopHandle>:$loops);
    let results = (outs LoopHandle:$fused_loop);
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` ($loops^ `:` type($loops))? `)` attr-dict `->` type($fused_loop)
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
    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, DefaultValuedAttr<UI32Attr,"0">:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` $loop `:` type($loop) `,` $factor `)` attr-dict
    }];
}

def HeteroCL_PipelineOp : HeteroCL_Op<"pipeline"> 
{
    let summary = "pipeline";
    let description = [{
        pipeline
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, DefaultValuedAttr<UI32Attr, "1">:$ii);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` $loop `:` type($loop) `,` $ii `)` attr-dict
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

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `:` type($stage) `,` $loop `:` type($loop) `)` attr-dict
    }];
}

def HeteroCL_PartitionOp : HeteroCL_Op<"partition"> 
{
    let summary = "partition";
    let description = [{
        hcl. partition(target, partition_type=0, dim=0, factor=0)

        Partition a Tensor into smaller Tensors or even registers

        Users can specify the partition type, which includes Complete, Block, and Cyclic. The default type is Complete, which means we completely partition the specified dimension. If Block is specified, the tensor is partitioned into N blocks with equal size. The number N is specified by the factor. Otherwise, if Cyclic is specified, the elements of the tensor is partition in a cyclic manner. For example, if the factor is three, the 1st element will be assigned to the 1st partitioned tensor; the 2nd element will be assigned to the 2nd one; and so on. Finally, if Complete is specified, the factor will be ignored. If dim is set to 0, it means we partition all dimensions.

        Parameters
        target (Tensor) - The tensor to be partitioned

        partition_type ({Complete, Block, Cyclic}, optional) - The partition type

        dim (int, optional) - The dimension to be partitioned

        factor (int, optional) - The partition factor
    }];

    let arguments = (ins AnyMemRef:$target, PartitionKindEnum:$partition_kind, DefaultValuedAttr<UI32Attr, "0">:$dim, DefaultValuedAttr<UI32Attr, "0">:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $partition_kind `,` $dim `,` $factor `)` attr-dict
    }];
}

def HeteroCL_BufferAtOp : HeteroCL_Op<"buffer_at"> 
{
    let summary = "buffer_at";
    let description = [{
        buffer_at
    }];

    // TODO: Also need to add Stage
    let arguments = (ins AnyMemRef:$target, DefaultValuedAttr<I32Attr, "-1">:$axis);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $axis `)` attr-dict
    }];
}

#endif // HCL_OPS
