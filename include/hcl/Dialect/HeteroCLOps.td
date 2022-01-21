//===- HeteroCLOps.td - hcl dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_OPS
#define HCL_OPS

include "hcl/Dialect/HeteroCLDialect.td"
include "hcl/Dialect/HeteroCLTypes.td"
include "hcl/Dialect/HeteroCLAttrs.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def HeteroCL_CreateLoopHandleOp : HeteroCL_Op<"create_loop_handle"> 
{
    let summary = "loop handle";
    let description = [{
        Create a loop handle

        Parameters
        loop_name (String) - The name of a loop
    }];

    let arguments = (ins StrAttr:$loop_name);
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
       $loop_name attr-dict `:` type($result)
    }];
}

def HeteroCL_CreateStageHandleOp : HeteroCL_Op<"create_stage_handle"> 
{
    let summary = "stage handle";
    let description = [{
        Create a stage handle

        Parameters
        stage_name (String) - The name of a stage
    }];

    let arguments = (ins StrAttr:$stage_name);
    let results = (outs StageHandle:$result);
    let assemblyFormat = [{
       $stage_name attr-dict `:` type($result)
    }];
}

def HeteroCL_ReorderOp : HeteroCL_Op<"reorder">
{
    let summary = "reorder";
    let description = [{
        hcl.reorder(*args)

        Reorder the arguments in the specified order.

        Parameters
        args (list of IterVar) - The order to be ordered
    }];

    let arguments = (ins StageHandle:$stage, Variadic<LoopHandle>:$loops);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `,` ($loops^)? `)` attr-dict
    }];
}

def HeteroCL_SplitOp : HeteroCL_Op<"split"> 
{
    let summary = "split";
    let description = [{
        hcl.split(parent, factor=None, nparts=None, mode='transform')

        Split the stage either by factor providing outer scope, or both

        Parameters
        * parent (IterVar) - The parent iter var.
        * factor (Expr, optional) - The splitting factor
        * nparts (Expr, optional) - The number of outer parts.
        * mode (str, "transform" or "annotate") - “transform” mode changes the IR structure, “annotate” mode adds attributes.

        Returns
        * outer (IterVar) - The outer variable of iteration.
        * inner (IterVar) - The inner variable of iteration.
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, UI32Attr:$factor);
    let results = (outs LoopHandle:$outer, LoopHandle:$inner);
    let assemblyFormat = [{
        `(` $stage `,` $loop `,` $factor `)` attr-dict
    }];
    // https://mlir.llvm.org/docs/OpDefinitions/#custom-builder-methods
    // let builders = [];
}

def HeteroCL_TileOp : HeteroCL_Op<"tile"> 
{
    let summary = "tile";
    let description = [{
        hcl.tile(x_parent, y_parent, x_factor, y_factor)

        Perform tiling on two dimensions

        The final loop order from outmost to inner most are [x_outer, y_outer, x_inner, y_inner]

        Parameters
        x_parent (IterVar) - The original x dimension
        y_parent (IterVar) - The original y dimension
        x_factor (Expr) - The stride factor on x axis
        y_factor (Expr) - The stride factor on y axis

        Returns
        x_outer (IterVar) - Outer axis of x dimension
        y_outer (IterVar) - Outer axis of y dimension
        x_inner (IterVar) - Inner axis of x dimension
        p_y_inner (IterVar) - Inner axis of y dimension
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$x_loop, LoopHandle:$y_loop, UI32Attr:$x_factor, UI32Attr:$y_factor);
    let results = (outs LoopHandle:$x_outer, LoopHandle:$x_inner, LoopHandle:$y_outer, LoopHandle:$y_inner);
    let assemblyFormat = [{
        `(` $stage `,` $x_loop `,` $y_loop `,` $x_factor `,` $y_factor `)` attr-dict
    }];
}

def HeteroCL_FuseOp : HeteroCL_Op<"fuse"> 
{
    let summary = "fuse";
    let description = [{
        hcl.fuse(*args)

        Fuse multiple consecutive iteration variables into a single iteration variable.

        fused = fuse(...fuse(fuse(args[0], args[1]), args[2]),..., args[-1]) The order is from outer to inner.

        Parameters
        args (list of IterVars) - Itervars that proceeds each other

        Returns
        fused - The fused variable of iteration.
    }];

    let arguments = (ins StageHandle:$stage, Variadic<LoopHandle>:$loops);
    let results = (outs LoopHandle:$fused_loop);
    let assemblyFormat = [{
        `(` $stage `,` ($loops^)? `)` attr-dict
    }];
}

def HeteroCL_ComputeAtOp : HeteroCL_Op<"compute_at"> 
{
    let summary = "compute_at";
    let description = [{
        hcl.compute_at(parent, scope)

        Attach the stage at parent's scope

        Parameters
        * parent (_Stage) - The parent stage
        * scope (IterVar) - The loop scope t be attached to.
    }];

    let arguments = (ins StageHandle:$producer, StageHandle:$consumer, LoopHandle:$axis);
    let results = (outs );
    let assemblyFormat = [{
        `(` $producer `,` $consumer `,` $axis `)` attr-dict
    }];
}

def HeteroCL_UnrollOp : HeteroCL_Op<"unroll"> 
{
    let summary = "unroll";
    let description = [{
        hcl.unroll(var, factor=0)

        Unroll the iteration.

        Parameters
        * var (IterVar) - The iteration to be unrolled.
        * factor (Expr) - The unroll factor. Default value 0 means full unroll.
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr,"0">>:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `,` $loop (`,` $factor^)? `)` attr-dict
    }];
}

def HeteroCL_PipelineOp : HeteroCL_Op<"pipeline"> 
{
    let summary = "pipeline";
    let description = [{
        hcl.pipeline(var, initiation_interval=1)

        Pipeline the iteration.

        Parameters
        * var (IterVar) - The iteration to be pipelined.
        * initiation_interval (Expr) - The initiation interval in pipeline schedule. Default value is 1.
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr, "1">>:$ii);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `,` $loop (`,` $ii^)? `)` attr-dict
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
        hcl.parallel(var)

        Parallelize the iteration.

        Parameters
        var (IterVar) - The iteration to be parallelized.
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `,` $loop `)` attr-dict
    }];
}

def HeteroCL_PartitionOp : HeteroCL_Op<"partition"> 
{
    let summary = "partition";
    let description = [{
        hcl.partition(target, partition_type=0, dim=0, factor=0)

        Partition a Tensor into smaller Tensors or even registers

        Users can specify the partition type, which includes Complete, Block, and Cyclic. The default type is Complete, which means we completely partition the specified dimension. If Block is specified, the tensor is partitioned into N blocks with equal size. The number N is specified by the factor. Otherwise, if Cyclic is specified, the elements of the tensor is partition in a cyclic manner. For example, if the factor is three, the 1st element will be assigned to the 1st partitioned tensor; the 2nd element will be assigned to the 2nd one; and so on. Finally, if Complete is specified, the factor will be ignored. If dim is set to 0, it means we partition all dimensions.

        Parameters
        * target (Tensor) - The tensor to be partitioned
        * partition_type ({Complete, Block, Cyclic}, optional) - The partition type
        * dim (int, optional) - The dimension to be partitioned
        * factor (int, optional) - The partition factor
    }];

    let arguments = (ins AnyMemRef:$target,
                     DefaultValuedAttr<PartitionKindEnum, "PartitionKindEnum::CompletePartition">:$partition_kind,
                     DefaultValuedAttr<UI32Attr, "0">:$dim,
                     OptionalAttr<DefaultValuedAttr<UI32Attr, "0">>:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $partition_kind `,` $dim (`,` $factor^)? `)` attr-dict
    }];
}

def HeteroCL_ReuseAtOp : HeteroCL_Op<"reuse_at"> 
{
    let summary = "reuse_at";
    let description = [{
        hcl.reuse_at(target, parent, axis, name=None)

        Create a reuse buffer reusing the output of current stage

        This returns a new tensor representing the reuse buffer. A stage is also built correspondingly. The new stage will be a sub-stage of the parent stage under the specified axis. Thus, the axis must be inside the axis list of the parent stage.

        Parameters
        * target (Tensor) - The tensor whose values will be reused
        * parent (Stage) - The stage that reuses the output of the current stage
        * axis (IterVar) - The axis that generates the reuse values
        * name (string, optional) - The name of the reuse buffer
    }];

    let arguments = (ins StageHandle:$stage, AnyMemRef:$target, LoopHandle:$axis);
    let results = (outs AnyMemRef:$result);
    let assemblyFormat = [{
        `(` $stage `,` $target `:` type($target) `,` $axis `)` attr-dict `->` type($result)
    }];
}

def HeteroCL_BufferAtOp : HeteroCL_Op<"buffer_at"> 
{
    let summary = "buffer_at";
    let description = [{
        hcl.buffer_at(target, parent, axis)

        Create a write buffer at the axis

        This returns a new tensor representing the write buffer. A stage is also built correspondingly. The new stage will be a sub-stage of the parent stage under the specified axis. Thus, the axis must be inside the axis list of the parent stage.

        Parameters
        * target (Tensor) - The tensor whose values will be buffered
        * parent (Stage) - The stage that includes the target tensor
        * axis (IterVar) - The axis that generates the buffer
    }];

    let arguments = (ins StageHandle:$stage, AnyMemRef:$target, LoopHandle:$axis);
    let results = (outs AnyMemRef:$result);
    let assemblyFormat = [{
        `(` $stage `,` $target `:` type($target) `,` $axis `)` attr-dict `->` type($result)
    }];
}

// def HeteroCL_ToOp : HeteroCL_Op<"to"> 
// {
//     let summary = "to";
//     let description = [{
//         hcl.to(tensor, dst=None, src=None, axis=0,
//                mode=IO.DMA, fifo_depth=1, burst_len=-1)

//         Stream a list of Tensors to dst devices

//         Parameters
//         ----------
//         tensor (Tensor) - The tensor to be moved
//         dst (Device or Stage) - The destination of data movement
//         src (Device or Stage) - The source of data movement
//         axis (str or IterVar) - Move axis-th loop body to xcel scope
//         mode (str) - The modes of data movement (Stream, DMA, MMIO)
//             For inter-kernel data movemnet, only Stream is supported
//         fifo_depth (int) - The streaming channel depth
//             We leave an interface here to specify the FIFO depth
//             in the future we should be able to infer automatically
//     }];

//     let arguments = (ins AnyMemRef:$target, DeviceEnum:$device);
//     let results = (outs );
//     let assemblyFormat = [{
//         `(` $target `:` type($target) `,` $device `)` attr-dict
//     }];
// }

def HeteroCL_ToOp : HeteroCL_Op<"to"> 
{
    let summary = "to";
    let description = [{
        hcl.to(tensor, dst=None, src=None, axis=0,
               mode=IO.DMA, fifo_depth=1, burst_len=-1)

        Stream a list of Tensors to dst devices

        Parameters
        ----------
        tensor (Tensor) - The tensor to be moved
        dst (Device or Stage) - The destination of data movement
        src (Device or Stage) - The source of data movement
        axis (str or IterVar) - Move axis-th loop body to xcel scope
        mode (str) - The modes of data movement (Stream, DMA, MMIO)
            For inter-kernel data movemnet, only Stream is supported
        fifo_depth (int) - The streaming channel depth
            We leave an interface here to specify the FIFO depth
            in the future we should be able to infer automatically
    }];

    let arguments = (ins AnyMemRef:$target, StageHandle:$dst);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $dst `)` attr-dict
    }];
}

// Fixed-point operators
def AnyFixed : Type<CPred<"$_self.isa<hcl::FixedType>() || $_self.isa<hcl::UFixedType>()">, "fixed-point",
                    "hcl::AnyFixedType">;
def FixedLike: TypeConstraint<Or<[AnyFixed.predicate,
        VectorOf<[AnyFixed]>.predicate, TensorOf<[AnyFixed]>.predicate]>,
    "fixed-point-like">;

class FixedBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<HeteroCL_Dialect, mnemonic, traits # [NoSideEffect,
        SameOperandsAndResultType] #
        ElementwiseMappable.traits>,
    Arguments<(ins FixedLike:$lhs, FixedLike:$rhs)> {

  let results = (outs AnyType:$result);
}

def AddFixedOp : FixedBinaryOp<"add_fixed"> {
  let summary = "fixed point addition operation";
}

def SubFixedOp : FixedBinaryOp<"sub_fixed"> {
  let summary = "fixed point subtraction operation";
}

def MulFixedOp : FixedBinaryOp<"mul_fixed"> {
  let summary = "fixed point mulplication operation";
}

#endif // HCL_OPS
