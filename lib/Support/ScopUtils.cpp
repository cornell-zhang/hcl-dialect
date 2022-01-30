//===- ScopUtils.cc -------------------------------------------------*- C++ -*-===//
//
// This file implements some generic utility functions.
//
//===----------------------------------------------------------------------===//

#include "hcl/Support/ScopUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SetVector.h"

using namespace hcl;
using namespace mlir;
using namespace llvm;

void hcl::inferBlockArgs(Block *block, llvm::SetVector<Value> &args) {
  // Any value being used will be added to the set first.
  block->walk([&](Operation *op) {
    for (mlir::Value operand : op->getOperands())
      args.insert(operand);
  });

  // Then we remove them if they are actually defined by the operations within
  // the current block.
  block->walk([&](Operation *op) {
    for (mlir::Value result : op->getResults())
      args.remove(result);

    Block *curr = op->getBlock();
    for (mlir::Value blkArg : curr->getArguments())
      args.remove(blkArg);
  });
}
