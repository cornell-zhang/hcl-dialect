//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AnyWidthInteger Pass
// This pass is to support any-width integer input from numpy.
// The input program has any-width integer input/output arguments
// The output program has 64-bit integer input/output and casts
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {


void updateTopFunctionSignature(FuncOp &funcOp) {
  FunctionType functionType = funcOp.getType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types;
  for (const auto &argEn : llvm::enumerate(funcOp.getArguments()))
    arg_types.push_back(argEn.value().getType());

  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;

  for (Type t : result_types) {
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      // If result memref element type is integer
      // change it to i64 to be compatible with numpy
      if (et.isa<IntegerType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_result_types.push_back(memrefType.clone(newElementType));
      } else {
        new_result_types.push_back(memrefType);
      }
    }
  }

  for (Type t : arg_types) {
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      // If argument memref element type is integer
      // change it to i64 to be compatible with numpy
      if (et.isa<IntegerType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_arg_types.push_back(memrefType.clone(newElementType));
      } else {
        new_arg_types.push_back(memrefType);
      }
    }
  }
  
  // Update FuncOp's block argument types
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      if (MemRefType memrefType = argType.cast<MemRefType>()) {
        Type et = memrefType.getElementType();
        if (et.isa<IntegerType>()) {
          size_t width = 64;
          Type newType = IntegerType::get(funcOp.getContext(), width);
          Type newMemRefType = memrefType.clone(newType);
          block.getArgument(i).setType(newMemRefType);
        }
      }
    }
  }

  // Update function signature
  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), new_arg_types, new_result_types);
  funcOp.setType(newFuncType);
}


/// entry point
bool applyAnyWidthInteger(ModuleOp &mod) {
  // Find top-level function
  bool isFoundTopFunc = false;
  FuncOp* topFunc;
  for (FuncOp func : mod.getOps<FuncOp>()) {
    if (func->hasAttr("top")) {
      isFoundTopFunc = true;
      topFunc = &func;
      break;
    }
  }

  if (isFoundTopFunc && topFunc) {
    updateTopFunctionSignature(*topFunc);
  }

  return true;
}

} // namespace hcl
} // namespace mlir

namespace {

struct HCLAnyWidthIntegerTransformation
  : public AnyWidthIntegerBase<HCLAnyWidthIntegerTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyAnyWidthInteger(mod))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass() {
  return std::make_unique<HCLAnyWidthIntegerTransformation>();
}

} // namespace hcl
} // namespace mlir