//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Transforms/HostXcelSeparation.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Support/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

bool applyHostXcelSeparation(
    ModuleOp &host_mod, ModuleOp &xcel_mod, ModuleOp &extern_mod,
    std::map<std::string, std::string> &device_map /*stage, xcel*/,
    std::vector<std::string> &graph_roots,
    std::vector<std::string> &subgraph_inputs,
    std::vector<std::string> &subgraph_outputs) {
  // get stage map: name->func_op
  // & get xcel top function
  std::map<std::string, FuncOp> funcMap;
  FuncOp xcel_top;
  for (FuncOp func : xcel_mod.getOps<FuncOp>()) {
    if (func->hasAttr("top"))
      xcel_top = func;
    else
      funcMap[func.getName().str().substr(6)] = func; // Stage_xxx
  }
  // get host main function
  FuncOp host_top;
  for (auto host_func : host_mod.getOps<FuncOp>()) {
    if (host_func.getName().str() == "main") {
      host_top = host_func;
      break;
    }
  }
  // get extern top function
  FuncOp extern_top;
  for (auto extern_func : extern_mod.getOps<FuncOp>()) {
    if (extern_func.getName().str() == "top") {
      extern_top = extern_func;
      break;
    }
  }
  // get host and xcel insertion points
  auto host_retOp = *(host_top.getOps<ReturnOp>().begin());
  auto xcel_retOp = *(xcel_top.getOps<ReturnOp>().begin());

  // get definitions map: name->array
  std::map<std::string, Value> defMap;
  for (auto def : xcel_top.getOps<memref::AllocOp>()) {
    std::string name = def->getAttr("name").cast<StringAttr>().getValue().str();
    defMap[name] = def.getResult();
  }
  for (auto def : host_top.getOps<memref::AllocOp>()) {
    std::string name = def->getAttr("name").cast<StringAttr>().getValue().str();
    defMap[name] = def.getResult();
  }
  std::string input_names =
      xcel_top->getAttr("inputs").cast<StringAttr>().getValue().str();
  std::vector<std::string> args = split_names(input_names);
  std::string output_names =
      xcel_top->getAttr("outputs").cast<StringAttr>().getValue().str();
  size_t len_inputs = args.size();
  for (auto it : llvm::enumerate(xcel_top.getArguments())) {
    if (it.index() < len_inputs) {
      defMap[args[it.index()]] = it.value();
    } else {
      defMap[output_names] = it.value();
    }
  }
  // get xcel stage call functions
  std::map<std::string, CallOp> callMap;
  std::vector<CallOp> callOrder;
  for (auto callOp : xcel_top.getOps<CallOp>()) {
    callMap[callOp.getCallee().str().substr(6)] = callOp;
    callOrder.push_back(callOp);
  }

  // get call @top function in host main
  auto call_top = *(host_top.getOps<CallOp>().begin());
  // auto top_ret = *(std::prev(call_top.arg_operand_end()));
  // only support zero or one output for now
  assert(subgraph_outputs.size() <= 1);

  // construct host module
  // should traverse in topological order
  bool isAfterXcelCall = false;
  for (auto callOp : callOrder) {
    std::string stage_name = callOp.getCallee().str().substr(6);
    if (subgraph_outputs.size() > 0 && stage_name == subgraph_outputs[0])
      isAfterXcelCall = true;
    // move stage from xcel to host
    std::string device = device_map[stage_name];
    if (device == "CPU" && funcMap.count(stage_name) > 0) {
      funcMap[stage_name]->moveBefore(host_top);
      // move array declaration from xcel to host
      auto stage_ret = *(std::prev(callMap[stage_name].arg_operand_end()));
      if (!isAfterXcelCall) {
        // return value should have been allocated
        stage_ret.getDefiningOp()->moveBefore(call_top);
        // move call function from xcel to host
        callMap[stage_name]->moveBefore(call_top);
      } else {
        // return value should have been allocated
        stage_ret.getDefiningOp()->moveBefore(host_retOp);
        // move call function from xcel to host
        callMap[stage_name]->moveBefore(host_retOp);
      }
    }
  }

  // fix reference (the following API will check all the operands in that
  // operation)
  for (auto callOp : host_top.getOps<CallOp>()) {
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
  for (auto callOp : xcel_top.getOps<CallOp>()) {
    auto input_names =
        callOp->getAttr("inputs").cast<StringAttr>().getValue().str();
    std::vector<std::string> args = split_names(input_names);
    // replace all references
    for (auto it : llvm::enumerate(callOp.getArgOperands())) {
      if (it.index() == callOp.getArgOperands().size() - 1) // skip return
        break;
      std::string arg = args[it.index()];
      if (defMap.count(arg + "_xcel") > 0) {
        auto def = defMap[arg + "_xcel"];
        callOp->replaceUsesOfWith(it.value(), def);
      } else if (defMap.count(arg) > 0) {
        auto def = defMap[arg];
        callOp->replaceUsesOfWith(it.value(), def);
      }
    }
  }
  // fix xcel return
  if (subgraph_outputs.size() != 0) {
    for (auto output : subgraph_outputs) {
      auto stage_ret = *(std::prev(callMap[output].arg_operand_end()));
      OpBuilder builder(xcel_retOp);
      builder.create<ReturnOp>(xcel_retOp->getLoc(), stage_ret);
      break;
    }
  } else {
    OpBuilder builder(xcel_retOp);
    builder.create<ReturnOp>(xcel_retOp->getLoc());
    // erase block arguments
    SmallVector<unsigned> argIdx;
    // func.front() is a block
    unsigned numArgs = xcel_top.front().getNumArguments();
    for (unsigned i = 0; i < numArgs; ++i)
      argIdx.push_back(i);
    xcel_top.front().eraseArguments(argIdx);
    // remove the host call
    call_top.erase();
  }
  // move stages to extern module
  for (auto func_op : xcel_mod.getOps<FuncOp>()) {
    if (func_op->hasAttr("systolic")) {
      func_op->moveBefore(extern_top);
      break; // only support placement for one kernel
    }
  }
  extern_top.erase();
  // remove original output
  xcel_retOp.erase();
  // set module name
  host_mod.setName("host");
  xcel_mod.setName("xcel");
  extern_mod.setName("extern");
  return true;
}

} // namespace hcl
} // namespace mlir