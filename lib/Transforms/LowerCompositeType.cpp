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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

void deadStructConstructElimination(FuncOp &func) {
  SmallVector<Operation *, 8> structConstructOps;
  func.walk([&](Operation *op) {
    if (auto structConstructOp = dyn_cast<StructConstructOp>(op)) {
      structConstructOps.push_back(structConstructOp);
    }
  });
  std::reverse(structConstructOps.begin(), structConstructOps.end());
  for (auto op : structConstructOps) {
    auto structValue = op->getResult(0);
    if (structValue.use_empty()) {
      op->erase();
    }
  }
}

void lowerStructType(FuncOp &func) {

  SmallVector<Operation *, 10> structGetOps;
  func.walk([&](Operation *op) {
    if (auto structGetOp = dyn_cast<StructGetOp>(op)) {
      structGetOps.push_back(structGetOp);
    }
  });

  for (auto op : structGetOps) {
    // Collect info from structGetOp
    auto structGetOp = dyn_cast<StructGetOp>(op);
    Value struct_value = structGetOp->getOperand(0);
    Value struct_field = structGetOp->getResult(0);
    auto index = structGetOp.index();

    // The defOp can be either a StructConstructOp or
    // a load from a memref.
    // Load: we are operating on a memref of struct
    // Construct: we are operating on a struct value
    Operation *defOp = struct_value.getDefiningOp();
    if (auto affine_load = dyn_cast<AffineLoadOp>(defOp)) {
      // defOp is loadOp from memref
      // Note: the idea to lower struct from memref is to
      // first create a memref for each struct field, and then
      // add store operation to those memrefs right after the
      // struct construction. With that, we can replace
      // struct get with loading from field memrefs.

      // Step1: create memref for each field
      Value struct_memref = affine_load.memref();
      OpBuilder builder(struct_memref.getDefiningOp());
      StructType struct_type = struct_value.getType().cast<StructType>();
      Location loc = op->getLoc();
      SmallVector<Value, 4> field_memrefs;
      for (Type field_type : struct_type.getElementTypes()) {
        MemRefType newMemRefType = struct_memref.getType()
                                       .cast<MemRefType>()
                                       .clone(field_type)
                                       .cast<MemRefType>();
        Value field_memref =
            builder.create<memref::AllocOp>(loc, newMemRefType);
        field_memrefs.push_back(field_memref);
      }

    } else if (auto structConstructOp = dyn_cast<StructConstructOp>(defOp)) {
      // defOp is a struct construction op
      Value replacement = defOp->getOperand(index);
      struct_field.replaceAllUsesWith(replacement);
      op->erase();
    } else {
      llvm_unreachable("unexpected defOp for structGetOp");
    }
  }

  // Run DCE after all struct get is folded
  deadStructConstructElimination(func);
}

/// Pass entry point
bool applyLowerCompositeType(ModuleOp &mod) {
  for (FuncOp func : mod.getOps<FuncOp>()) {
    lowerStructType(func);
    llvm::outs() << func << "\n";
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