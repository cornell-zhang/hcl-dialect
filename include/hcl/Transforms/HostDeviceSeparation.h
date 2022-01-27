//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HETEROCL_HOSTDEVICESEPARATION_H
#define HETEROCL_HOSTDEVICESEPARATION_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

#include <map>

namespace mlir {
namespace hcl {

bool applyHostDeviceSeparation(ModuleOp &host_mod, ModuleOp &device_mod,
                               std::map<std::string, std::string> &device_map,
                               std::vector<std::string> &subgraph_inputs,
                               std::vector<std::string> &subgraph_outputs);

} // namespace hcl
} // namespace mlir

#endif // HETEROCL_HOSTDEVICESEPARATION_H