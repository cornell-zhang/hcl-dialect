#ifndef HETEROCL_PASSES_H
#define HETEROCL_PASSES_H
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

bool applyLoopTransformation(FuncOp &f);

void registerHCLLoopTransformationPass();
std::unique_ptr<mlir::Pass> createHCLLoopTransformationPass();

} // namespace hcl
} // namespace mlir

#endif // HETEROCL_PASSES_H