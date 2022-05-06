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

void deadMemRefAllocElimination(FuncOp &func) {
  SmallVector<Operation *, 8> memRefAllocOps;
  func.walk([&](Operation *op) {
    if (auto memRefAllocOp = dyn_cast<memref::AllocOp>(op)) {
      memRefAllocOps.push_back(memRefAllocOp);
    }
  });
  std::reverse(memRefAllocOps.begin(), memRefAllocOps.end());
  for (auto op : memRefAllocOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
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

  std::map<llvm::hash_code, SmallVector<Value, 8>> structMemRef2fieldMemRefs;

  for (auto op : structGetOps) {
    // Collect info from structGetOp
    auto structGetOp = dyn_cast<StructGetOp>(op);
    Value struct_value = structGetOp->getOperand(0);
    Value struct_field = structGetOp->getResult(0);
    Location loc = op->getLoc();
    auto index = structGetOp.index();

    // This flag indicates whether we can erase
    // struct construct and relevant ops.
    bool erase_struct_construct = false;

    // The defOp can be either a StructConstructOp or
    // a load from a memref.
    // Load: we are operating on a memref of struct
    // Construct: we are operating on a struct value
    Operation *defOp = struct_value.getDefiningOp();
    if (auto affine_load = dyn_cast<AffineLoadOp>(defOp)) {
      // Case 1: defOp is loadOp from memref
      // Note: the idea to lower struct from memref is to
      // first create a memref for each struct field, and then
      // add store operation to those memrefs right after the
      // struct construction. With that, we can replace
      // struct get with loading from field memrefs.

      // Step1: create memref for each field
      Value struct_memref = affine_load.memref();
      // Try to find field_memrefs associated with this struct_memref
      SmallVector<Value, 4> field_memrefs;
      auto it = structMemRef2fieldMemRefs.find(hash_value(struct_memref));
      if (it == structMemRef2fieldMemRefs.end()) {
        // Create a memref for each field
        OpBuilder builder(struct_memref.getDefiningOp());
        StructType struct_type = struct_value.getType().cast<StructType>();

        for (Type field_type : struct_type.getElementTypes()) {
          MemRefType newMemRefType = struct_memref.getType()
                                         .cast<MemRefType>()
                                         .clone(field_type)
                                         .cast<MemRefType>();
          Value field_memref =
              builder.create<memref::AllocOp>(loc, newMemRefType);
          field_memrefs.push_back(field_memref);
        }
        structMemRef2fieldMemRefs.insert(
            std::make_pair(hash_value(struct_memref), field_memrefs));
        erase_struct_construct = true;
      } else {
        field_memrefs.append(it->second);
        erase_struct_construct = false;
      }

      // Step2: add store to each field memref
      for (auto &use : struct_memref.getUses()) {
        if (auto storeOp = dyn_cast<AffineStoreOp>(use.getOwner())) {
          // Find a storeOp to the struct memref, we add
          // store to each field memref here.
          OpBuilder builder(storeOp);
          for (const auto &field_memref_en : llvm::enumerate(field_memrefs)) {
            auto field_memref = field_memref_en.value();
            auto field_index = field_memref_en.index();
            // Find the struct_construct op
            auto struct_construct_op = dyn_cast<StructConstructOp>(
                storeOp.getOperand(0).getDefiningOp());
            builder.create<AffineStoreOp>(
                loc, struct_construct_op.getOperand(field_index), field_memref,
                storeOp.indices());
          }
          // erase the storeOp that stores to the struct memref
          if (erase_struct_construct) {
            storeOp.erase();
          }
          break;
        }
      }

      // Step3: replace structGetOp with load from field memrefs
      OpBuilder load_builder(op);
      Value loaded_field = load_builder.create<AffineLoadOp>(
          loc, field_memrefs[index], affine_load.indices());
      struct_field.replaceAllUsesWith(loaded_field);
      op->erase();

      // erase the loadOp from struct_memref
      defOp->erase();

    } else if (auto structConstructOp = dyn_cast<StructConstructOp>(defOp)) {
      // Case 2: defOp is a struct construction op
      Value replacement = defOp->getOperand(index);
      struct_field.replaceAllUsesWith(replacement);
      op->erase();
    } else {
      llvm_unreachable("unexpected defOp for structGetOp");
    }
  }

  // Run DCE after all struct get is folded
  deadStructConstructElimination(func);
  deadMemRefAllocElimination(func);
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