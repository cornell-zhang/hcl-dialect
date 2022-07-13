//===- HeteroCLOps.cpp - hcl dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hcl/Dialect/HeteroCLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace hcl {

static ParseResult parseCustomizationOp(OpAsmParser &parser,
                                        OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(CustomizationOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_interface_impl::printFunctionOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

static LogicalResult verify(CustomizationOp op) {
  // If this function is external there is nothing to do.
  if (op.isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = op.getType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CmpFixedOp
//===----------------------------------------------------------------------===//

static void buildCmpFixedOp(OpBuilder &build, OperationState &result,
                            CmpFixedPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  result.addAttribute(CmpFixedOp::getPredicateAttrName(),
                      build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

} // namespace hcl
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hcl/Dialect/HeteroCLOps.cpp.inc"
