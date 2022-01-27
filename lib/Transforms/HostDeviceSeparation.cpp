//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//h

#include "hcl/Transforms/HostDeviceSeparation.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Support/Utils.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

bool applyHostDeviceSeparation(
    ModuleOp &host_mod, ModuleOp &device_mod,
    std::map<std::string, std::string> &device_map /*stage, device*/,
    std::vector<std::string> &subgraph_inputs,
    std::vector<std::string> &subgraph_outputs) {
  // get stage map: name->func_op
  std::map<std::string, FuncOp> funcMap;
  FuncOp top_func;
  for (FuncOp func : device_mod.getOps<FuncOp>()) {
    if (func->hasAttr("top"))
      top_func = func;
    else
      funcMap[func.getName().str().substr(6)] = func; // Stage_xxx
  }
  // get host main function
  FuncOp host_main;
  for (auto host_func : host_mod.getOps<FuncOp>()) {
    if (host_func.getName().str() == "main") {
      host_main = host_func;
      break;
    }
  }
  auto host_retOp = *(host_main.getOps<ReturnOp>().begin());
  auto device_retOp = *(top_func.getOps<ReturnOp>().begin());
  // get device stage call functions
  std::map<std::string, CallOp> callMap;
  for (auto callOp : top_func.getOps<CallOp>()) {
    callMap[callOp.getCallee().str().substr(6)] = callOp;
  }
  // get call @top function in host main
  auto call_top = *(host_main.getOps<CallOp>().begin());
  auto top_ret = *(std::prev(call_top.arg_operand_end()));
  // construct host module
  for (auto item : device_map) {
    std::string stage_name = item.first;
    std::string device = item.second;
    // move stage from device to host
    llvm::errs() << stage_name << " " << device << "\n";
    if (device == "CPU" && funcMap.count(stage_name) > 0) {
      funcMap[stage_name]->moveBefore(host_main);
      // move array declaration from device to host
      auto stage_ret = *(std::prev(callMap[stage_name].arg_operand_end()));
      // return value should have been allocated
      stage_ret.getDefiningOp()->moveBefore(host_retOp);
      // move call function from device to host
      callMap[stage_name]->moveBefore(host_retOp);
      // fix reference (the following API will check all the operands in that operation)
      callMap[stage_name]->replaceUsesOfWith(*callMap[stage_name].arg_operand_begin(),top_ret);
    }
  }
  // fix device output
  // only support one output for now
  assert(subgraph_outputs.size() == 1);
  for (auto output : subgraph_outputs) {
    auto stage_ret = *(std::prev(callMap[output].arg_operand_end()));
    OpBuilder builder(device_retOp);
    builder.create<ReturnOp>(device_retOp->getLoc(), stage_ret);
    break;
  }
  // remove original output
  device_retOp.erase();
  // set module name
  host_mod.setName("host");
  device_mod.setName("device");
  return true;
}

} // namespace hcl
} // namespace mlir