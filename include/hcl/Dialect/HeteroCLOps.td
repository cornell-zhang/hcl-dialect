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

include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

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

def HeteroCL_ThreadBindOp : HeteroCL_Op<"bind">
{
    let summary = "bind";
    let description = [{
        hcl.bind(var, dim=1)

        Bind the iteration to a specific NDRange dimension.

        Parameters
        * var (IterVar) - The iteration to be mapped.
        * dim (Expr) - The NDRange dimension to be binded to the iteration.
    }];

    let arguments = (ins StageHandle:$stage, LoopHandle:$loop, NDRangeDimKindEnum:$dim);
    let results = (outs );
    let assemblyFormat = [{
        `(` $stage `,` $loop `,` $dim `)` attr-dict
    }];
    let builders = [
        OpBuilder<(ins "hcl::LoopHandleType":$loop,
                   CArg<"uint64_t", "1">:$dim)>
    ];
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

def HeteroCL_ReshapeOp : HeteroCL_Op<"reshape"> 
{
    let summary = "reshape";
    let description = [{
        hcl.reshape(target)

        Reshape the tensor by modifying the memref
    }];

    let arguments = (ins AnyMemRef:$target);
    let results = (outs AnyMemRef:$output);
    let assemblyFormat = [{
        $target `:` type($target) `to` type($output) attr-dict
    }];
}

def HeteroCL_InterKernelToOp : HeteroCL_Op<"inter_kernel_to"> 
{
    let summary = "inter-kernel data placement";
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

    let arguments = (ins AnyMemRef:$target, StageHandle:$dst,
                         OptionalAttr<DefaultValuedAttr<I32Attr,"-1">>:$fifo_depth);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $dst (`,` $fifo_depth^)? `)` attr-dict
    }];
}

def HeteroCL_OutlineOp : HeteroCL_Op<"outline"> 
{
    let summary = "outline";
    let description = [{
        hcl.outline(stage)

        Outline a stage as a function
    }];

    let arguments = (ins StageHandle:$stage);
    let results = (outs ); //(outs FunctionType)
    let assemblyFormat = [{
        `(` $stage `)` attr-dict
    }];
}

// Fixed-point operators
def AnyFixed : Type<CPred<"$_self.isa<hcl::FixedType>() || $_self.isa<hcl::UFixedType>()">, "fixed-point",
                    "hcl::AnyFixedType">;
def FixedLike: TypeConstraint<Or<[AnyFixed.predicate,
        VectorOf<[AnyFixed]>.predicate, TensorOf<[AnyFixed]>.predicate]>,
    "fixed-point-like">;

class FixedBinaryOp<string mnemonic, list<Trait> traits = []> :
    Op<HeteroCL_Dialect, mnemonic, traits # [SameOperandsAndResultType]>,
    Arguments<(ins FixedLike:$lhs, FixedLike:$rhs)>,
    Results<(outs FixedLike:$result)>;

def AddFixedOp : FixedBinaryOp<"add_fixed"> {
  let summary = "fixed point addition operation";
}

def SubFixedOp : FixedBinaryOp<"sub_fixed"> {
  let summary = "fixed point subtraction operation";
}

def MulFixedOp : FixedBinaryOp<"mul_fixed"> {
  let summary = "fixed point mulplication operation";
}

def DivFixedOp : FixedBinaryOp<"div_fixed"> {
  let summary = "fixed point division operation";
}

def CmpFixedOp : HeteroCL_Op<"cmp_fixed", [NoSideEffect, SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">] # ElementwiseMappable.traits> {
  let summary = "fixed-point comparison operation";

  let arguments = (ins
      CmpFixedPredicateAttr:$predicate,
      FixedLike:$lhs,
      FixedLike:$rhs
  );
  let results = (outs BoolLike:$result);

  let builders = [
    OpBuilder<(ins "CmpFixedPredicate":$predicate, "Value":$lhs,
                 "Value":$rhs), [{
      mlir::hcl::buildCmpFixedOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpFixedPredicate getPredicateByName(StringRef name);

    CmpFixedPredicate getPredicate() {
      return (CmpFixedPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def MinFixedOp : FixedBinaryOp<"min_fixed"> {
  let summary = "fixed point minimum operation";
}

def MaxFixedOp : FixedBinaryOp<"max_fixed"> {
  let summary = "fixed point maximum operation";
}

def GetGlobalFixedOp : HeteroCL_Op<"get_global_fixed"> {
    let summary = "generates a fixed point memref from a int global memref";
    let arguments = (ins FlatSymbolRefAttr:$global);
    let results = (outs AnyStaticShapeMemRef:$result);
    let assemblyFormat = [{
        $global `:` type($result) attr-dict
    }];
}

// Bitwise operators
def GetIntBitOp : HeteroCL_Op<"get_bit"> {
    let summary = "get bit of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$index);
    let results = (outs BoolLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $index `)` attr-dict `->` type($result)
    }];
}

def SetIntBitOp : HeteroCL_Op<"set_bit"> {
    let summary = "set bit of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$index, BoolLike:$val);
    let results = (outs );
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $index `,` $val `:` type($val) `)` attr-dict
    }];
}

def GetIntSliceOp : HeteroCL_Op<"get_slice"> {
    let summary = "get slice of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$hi, Builtin_Index:$lo);
    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $hi `,` $lo `)` attr-dict `->` type($result)
    }];
}

def SetIntSliceOp : HeteroCL_Op<"set_slice"> {
    let summary = "set slice of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$hi, Builtin_Index:$lo, SignlessIntegerLike:$val);
    let results = (outs );
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $hi `,` $lo `,` $val `:` type($val) `)` attr-dict
    }];
}

def BitReverseOp : HeteroCL_Op<"bit_reverse", [SameOperandsAndResultType]> {
    let summary = "reverse bits of an integer";
    let arguments = (ins SignlessIntegerLike:$num);
    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `)` attr-dict
    }];
}

def PrintOp : HeteroCL_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input memref, and produces
    no results.
  }];

  // The print operation takes an input memref to print.
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input)";
}

def FixedToFloatOp : HeteroCL_Op<"fixed_to_float"> {
  let summary = "fixed to float cast operation";
  let arguments = (ins FixedLike:$input);
  let results = (outs FloatLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def FloatToFixedOp : HeteroCL_Op<"float_to_fixed"> {
  let summary = "float to fixed cast operation";
  let arguments = (ins FixedLike:$input);
  let results = (outs FloatLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def IntToFixedOp : HeteroCL_Op<"int_to_fixed"> {
  let summary = "int to fixed cast operation";
  let arguments = (ins SignlessIntegerLike:$input);
  let results = (outs FixedLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def FixedToIntOp : HeteroCL_Op<"fixed_to_int"> {
  let summary = "fixed to int cast operation";
  let arguments = (ins FixedLike:$input);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def FixedToFixedOp : HeteroCL_Op<"fixed_to_fixed"> {
  let summary = "Cast operation from one fixed point type to another";
  let arguments = (ins FixedLike:$input);
  let results = (outs FixedLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def StructConstructOp : HeteroCL_Op<"struct_construct", [NoSideEffect]> {
    let summary = "struct construct";
    let description = [{
        The "struct_construct" builtin operation constructs a struct from
        a list of values.
    }];
    let arguments = (ins Variadic<AnyType>:$input);
    let results = (outs Struct:$output);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `->` type($output)
    }];
}


def StructGetOp : HeteroCL_Op<"struct_get", [NoSideEffect]> {
    let summary = "struct get";
    let description = [{
        The "struct_get" operation gets a field from a struct.
    }];
    let arguments = (ins Struct:$input, I64Attr:$index);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
        $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$index)>
    ];

}

def LogicalAndOp : HeteroCL_Op<"logical_and"> {
    let summary = "logical and operation";
    let arguments = (ins Variadic<AnyType>:$input);
    let results = (outs BoolLike:$result);
    let assemblyFormat = [{
        ($input^)? attr-dict `:` type($input) `->` type($result)
    }];
}

def LogicalOrOp : HeteroCL_Op<"logical_or"> {
    let summary = "logical or operation";
    let arguments = (ins Variadic<AnyType>:$input);
    let results = (outs BoolLike:$result);
    let assemblyFormat = [{
        ($input^)? attr-dict `:` type($input) `->` type($result)
    }];
}

#endif // HCL_OPS
