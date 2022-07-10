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

#include "mlir/Dialect/MemRef/IR/MemRef.h"


using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

void removeStrideMap(FuncOp &func) {
  SmallVector<Operation *, 8> allocOps;
  func.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(alloc);
    }
  });

  for (auto op : allocOps) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType().cast<MemRefType>();
    auto memRefMaps = memRefType.getLayout();
    if (memRefMaps.getAffineMap().isIdentity() || memRefMaps.getAffineMap().isEmpty()) {
      continue;
    }
    auto newMemRefType = MemRefType::get(
        memRefType.getShape(), memRefType.getElementType());
    op->getResult(0).setType(newMemRefType);
  }


  FunctionType functionType = funcOp.getType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types;
  for (const auto &argEn : llvm::enumerate(funcOp.getArguments()))
    arg_types.push_back(argEn.value().getType());
  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;
  for (auto result_type : result_types) {
    if (result_type.isa<MemRefType>()) {
       MemRefType new_result_type = memRefType::get(
           result_type.cast<MemRefType>().getShape(),
           result_type.cast<MemRefType>().getElementType());
      new_result_types.push_back(result_type);
    }
  }
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