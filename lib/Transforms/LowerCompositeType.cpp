//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LowerCompositeType Pass
// This file defines the lowering of composite types such as structs.
// This pass is separated from HCLToLLVM because it could be used
// in other backends as well, such as HLS backend.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

  void lowerStructType(FuncOp &func) {
    SmallVector<Operation *, 10> structGetOps;
    func.walk([&](Operation *op) {
      if (auto structGetOp = dyn_cast<StructGetOp>(op)) {
        structGetOps.push_back(structGetOp);
      }
    });

    for (auto op : structGetOps) {
      auto structGetOp = dyn_cast<StructGetOp>(op);
      Value struct_value = structGetOp->getOperand(0);
      Value struct_field = structGetOp->getResult(0);
      auto index = structGetOp.index();
      Operation* defOp = struct_value.getDefiningOp();
      Value replacement = defOp->getOperand(index);
      struct_field.replaceAllUsesWith(replacement);
      op->erase();
    }

  }


  /// Pass entry point
  bool applyLowerCompositeType(ModuleOp &mod) {
    for (FuncOp func : mod.getOps<FuncOp>()) {
      lowerStructType(func);
    }
    return true;
  }
} // namespace hcl
} // namespace mlir

namespace {
struct HCLLowerCompositeTypeTransformation
    : public LowerCompositeTypeBase<HCLLowerCompositeTypeTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerCompositeType(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass() {
  return std::make_unique<HCLLowerCompositeTypeTransformation>();
}
} // namespace hcl
} // namespace mlir