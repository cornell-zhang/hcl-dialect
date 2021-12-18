#ifndef HETEROCL_PASSES_H
#define HETEROCL_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "hcl/HeteroCLOps.h"

namespace mlir {
namespace hcl {

void registerHCLLoopTransformationPass();
std::unique_ptr<mlir::Pass> createHCLLoopTransformationPass();

} // namespace hcl
} // namespace mlir

#endif // HETEROCL_PASSES_H