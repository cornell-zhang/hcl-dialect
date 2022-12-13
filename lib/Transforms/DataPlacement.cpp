//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include <map>

#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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


/// Pass entry point
bool applyDataPlacement(ModuleOp &module) { 
  // try creating a new module
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  builder.create<ModuleOp>(module.getLoc());
//   auto newModule = ModuleOp::create(module.getLoc());
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