//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <map>
#include <set>

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

class Node {
  // Member variables
  Operation *op;
  int device;
  std::vector<Node *> upstream;
  std::vector<Node *> downstream;
 public:
  Node(Operation *op) : op(op) {}
  void addUpstream(Node *node) { upstream.push_back(node); }
  void addDownstream(Node *node) { downstream.push_back(node); } 
};

class DataFlowGraph {
  // Member variables
  std::vector<Node *> nodes;
 public:
  void addNode(Node *node) { nodes.push_back(node); }
  void addEdge(Node *src, Node *dst) {
    src->addDownstream(dst);
    dst->addUpstream(src);
  }
};

void getAllLoadedMemRefs(Operation *op, std::set<Operation *> &memRefs) {
  SmallVector<Operation *, 8> loadOps;
  op->walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op)) {
      loadOps.push_back(op);
    } else if (isa<memref::LoadOp>(op)) {
      loadOps.push_back(op);
    }
  });

  // add memrefs to the set
  for (auto loadOp : loadOps) {
    memRefs.insert(loadOp->getOperand(0).getDefiningOp());
  }
}

void getAllStoredMemRefs(Operation *op, std::set<Operation *> &memRefs) {
  SmallVector<Operation *, 8> storeOps;
  op->walk([&](Operation *op) {
    if (isa<AffineStoreOp>(op)) {
      storeOps.push_back(op);
    } else if (isa<memref::StoreOp>(op)) {
      storeOps.push_back(op);
    }
  });

  // add memrefs to the set
  for (auto storeOp : storeOps) {
    memRefs.insert(storeOp->getOperand(1).getDefiningOp());
  }
}

/// Pass entry point
bool applyDataPlacement(ModuleOp &module) { 
  
  std::map<Operation *, std::set<Operation *>> loopConsumedMemRefs;
  std::map<Operation *, std::set<Operation *>> loopProducedMemRefs;

  // get all the loops
  std::vector<Operation *> loops;
  module.walk([&](Operation *op) {
    if (isa<AffineForOp>(op)) {
      loops.push_back(op);
    } else if (isa<scf::ForOp>(op)) {
      loops.push_back(op);
    }
  });

  // get all the memrefs consumed and produced by each loop
  for (auto loop : loops) {
    // add an empty set for each loop
    loopConsumedMemRefs.insert(std::make_pair(loop, std::set<Operation *>()));
    loopProducedMemRefs.insert(std::make_pair(loop, std::set<Operation *>()));
    getAllLoadedMemRefs(loop, loopConsumedMemRefs[loop]);
    getAllStoredMemRefs(loop, loopProducedMemRefs[loop]);
  }
  
  
  
  // try creating a new module
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  builder.create<ModuleOp>(module.getLoc());
  // move all ops in the old module to the new module
//   newModule.getBody()->getOperations().splice(
//         newModule.getBody()->begin(), module.getBody()->getOperations());
  return true;
}

} // namespace hcl
} // namespace mlir

namespace {
struct HCLDataPlacementTransformation
    : public DataPlacementBase<HCLDataPlacementTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyDataPlacement(mod)) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {
std::unique_ptr<OperationPass<ModuleOp>> createDataPlacementPass() {
  return std::make_unique<HCLDataPlacementTransformation>();
}
} // namespace hcl
} // namespace mlir