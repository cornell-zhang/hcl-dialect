//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MoveReturnToInput Pass
// This pass is to support multiple return values for LLVM backend.
// The input program may have multiple return values
// The output program has no return values, all the return values 
// is moved to the input argument list
//===----------------------------------------------------------------------===//


#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

void moveReturnToInput(FuncOp& funcOp) {
  FunctionType functionType = funcOp.getType();
  SmallVector<Type, 4> resTypes = llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 4> argTypes;
  for (auto& arg : funcOp.getArguments()) {
    argTypes.push_back(arg.getType());
  }
  // Append resTypes to argTypes and clear resTypes
  argTypes.insert(std::end(argTypes), std::begin(resTypes), std::end(resTypes));
  resTypes.clear();
  // Update function signature
  FunctionType newFuncType = 
    FunctionType::get(funcOp.getContext(), argTypes, resTypes);
  funcOp.setType(newFuncType);
}

/// entry point
bool applyMoveReturnToInput(ModuleOp &mod) {
  // Find top-level function
  bool isFoundTopFunc = false;
  FuncOp *topFunc;
  for (FuncOp func : mod.getOps<FuncOp>()) {
    if (func->hasAttr("top")) {
      isFoundTopFunc = true;
      topFunc = &func;
      break;
    }
  }

  if (isFoundTopFunc && topFunc) {
    moveReturnToInput(*topFunc);
  }
  return true;
}

} // namespace hcl
} // namespace mlir

namespace {

struct HCLMoveReturnToInputTransformation
    : public MoveReturnToInputBase<HCLMoveReturnToInputTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyMoveReturnToInput(mod))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createMoveReturnToInputPass() {
  return std::make_unique<HCLMoveReturnToInputTransformation>();
}

} // namespace hcl
} // namespace mlir