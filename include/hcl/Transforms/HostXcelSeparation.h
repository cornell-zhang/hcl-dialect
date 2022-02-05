//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HETEROCL_HOSTXCELSEPARATION_H
#define HETEROCL_HOSTXCELSEPARATION_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

#include <map>

namespace mlir {
namespace hcl {

bool applyHostXcelSeparation(ModuleOp &host_mod, ModuleOp &xcel_mod,
                             ModuleOp &extern_mod,
                             std::map<std::string, std::string> &device_map,
                             std::vector<std::string> &graph_roots,
                             std::vector<std::string> &subgraph_inputs,
                             std::vector<std::string> &subgraph_outputs);

} // namespace hcl
} // namespace mlir

#endif // HETEROCL_HOSTXCELSEPARATION_H