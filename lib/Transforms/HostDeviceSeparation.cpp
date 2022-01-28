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

std::vector<std::string> split_names(const std::string &arg_names) {
  std::stringstream ss(arg_names);
  std::vector<std::string> args;
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    args.push_back(substr);
  }
  return args;
}

bool applyHostDeviceSeparation(
    ModuleOp &host_mod, ModuleOp &device_mod,
    std::map<std::string, std::string> &device_map /*stage, device*/,
    std::vector<std::string> &graph_roots,
    std::vector<std::string> &subgraph_inputs,
    std::vector<std::string> &subgraph_outputs) {
  // get stage map: name->func_op
  // & get device top function
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
  // get host and device insertion points
  auto host_retOp = *(host_main.getOps<ReturnOp>().begin());
  auto device_retOp = *(top_func.getOps<ReturnOp>().begin());

  // get definitions map: name->array
  std::map<std::string, Value> defMap;
  for (auto def : top_func.getOps<memref::AllocOp>()) {
    std::string name = def->getAttr("name").cast<StringAttr>().getValue().str();
    defMap[name] = def.getResult();
  }
  for (auto def : host_main.getOps<memref::AllocOp>()) {
    std::string name = def->getAttr("name").cast<StringAttr>().getValue().str();
    defMap[name] = def.getResult();
  }
  std::string input_names =
      top_func->getAttr("inputs").cast<StringAttr>().getValue().str();
  std::vector<std::string> args = split_names(input_names);
  std::string output_names =
      top_func->getAttr("outputs").cast<StringAttr>().getValue().str();
  size_t len_inputs = args.size();
  for (auto it : llvm::enumerate(top_func.getArguments())) {
    if (it.index() < len_inputs) {
      defMap[args[it.index()]] = it.value();
    } else {
      defMap[output_names] = it.value();
    }
  }
  // get device stage call functions
  std::map<std::string, CallOp> callMap;
  std::vector<CallOp> callOrder;
  for (auto callOp : top_func.getOps<CallOp>()) {
    callMap[callOp.getCallee().str().substr(6)] = callOp;
    callOrder.push_back(callOp);
  }

  // get call @top function in host main
  auto call_top = *(host_main.getOps<CallOp>().begin());
  auto top_ret = *(std::prev(call_top.arg_operand_end()));
  // only support one output for now
  assert(subgraph_outputs.size() == 1);

  // construct host module
  // should traverse in topological order
  bool isBack = false;
  for (auto callOp : callOrder) {
    std::string stage_name = callOp.getCallee().str().substr(6);
    if (stage_name == subgraph_outputs[0])
      isBack = true;
    // move stage from device to host
    std::string device = device_map[stage_name];
    if (device == "CPU" && funcMap.count(stage_name) > 0) {
      funcMap[stage_name]->moveBefore(host_main);
      // move array declaration from device to host
      auto stage_ret = *(std::prev(callMap[stage_name].arg_operand_end()));
      if (!isBack) {
        // return value should have been allocated
        stage_ret.getDefiningOp()->moveBefore(call_top);
        // move call function from device to host
        callMap[stage_name]->moveBefore(call_top);
      } else {
        // return value should have been allocated
        stage_ret.getDefiningOp()->moveBefore(host_retOp);
        // move call function from device to host
        callMap[stage_name]->moveBefore(host_retOp);
      }
    }
  }

  // fix reference (the following API will check all the operands in that
  // operation)
  for (auto callOp : host_main.getOps<CallOp>()) {
    auto input_names =
        callOp->getAttr("inputs").cast<StringAttr>().getValue().str();
    std::vector<std::string> args = split_names(input_names);
    // suppose only one output
    args.push_back(
        callOp->getAttr("outputs").cast<StringAttr>().getValue().str());
    // replace all references
    for (auto it : llvm::enumerate(callOp.getArgOperands())) {
      std::string arg = args[it.index()];
      if (defMap.count(arg + "_host") > 0) {
        auto def = defMap[arg + "_host"];
        callOp->replaceUsesOfWith(it.value(), def);
      } else if (defMap.count(arg) > 0) {
        auto def = defMap[arg];
        callOp->replaceUsesOfWith(it.value(), def);
      }
    }
  }
  for (auto callOp : top_func.getOps<CallOp>()) {
    auto input_names =
        callOp->getAttr("inputs").cast<StringAttr>().getValue().str();
    std::vector<std::string> args = split_names(input_names);
    // replace all references
    for (auto it : llvm::enumerate(callOp.getArgOperands())) {
      if (it.index() == callOp.getArgOperands().size() - 1) // skip return
        break;
      std::string arg = args[it.index()];
      if (defMap.count(arg + "_device") > 0) {
        auto def = defMap[arg + "_device"];
        callOp->replaceUsesOfWith(it.value(), def);
      } else if (defMap.count(arg) > 0) {
        auto def = defMap[arg];
        callOp->replaceUsesOfWith(it.value(), def);
      }
    }
  }
  // fix device return
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