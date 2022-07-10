//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
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

void removeStrideMap(FuncOp &func) {
}
   
/// Pass entry point
bool applyRemoveStrideMap(ModuleOp &module) {
  for (FuncOp func : module.getOps<FuncOp>()) {
    removeStrideMap(func);
  }
  return true;
}

} // namespace hcl
} // namespace mlir

namespace {
struct HCLRemoveStrideMapTransformation : public RemoveStrideMapBase<HCLRemoveStrideMapTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyRemoveStrideMap(mod)) {
        signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {
std::unique_ptr<OperationPass<ModuleOp>> createRemoveStrideMapPass() {
  return std::make_unique<HCLRemoveStrideMapTransformation>();
}
} // namespace hcl
} // namespace mlir