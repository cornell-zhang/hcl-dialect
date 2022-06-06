//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitMlirGpu.h"
#include "hcl/Dialect/Visitor.h"
#include "hcl/Support/Utils.h"
#include "hcl/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"

using namespace mlir;
using namespace hcl;

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public HCLEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(HCLEmitterState &state) : HCLEmitterBase(state) {}

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  void emitInfoAndNewLine(Operation *op);
  void emitBlock(Block &block);
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  void emitFunction(FuncOp func);
};
} // namespace

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();
  os << "\n";
}

void ModuleEmitter::emitBlock(Block &block) {
  os << "\n=====BEGIN BLOCK EMIT=====\n";
  for (auto &op : block) {
    // if (ExprVisitor(*this).dispatchVisitor(&op))
    //   continue;

    // if (StmtVisitor(*this).dispatchVisitor(&op))
    //   continue;

    // emitError(&op, "can't be correctly emitted.");
    os << op;
  }
  os << "\n";
  os << "=====END BLOCKEMIT=====\n\n";

  // TODO: hard-code for now..
  indent();
  os << "// Register data to device\n";
  indent();
  os << "gpu.host_register %cast_'v[0-9]+' : type(memref<256xf32>)\n\n";
  indent();
  os << "// GPU Kernel definition\n";
  indent();
  os << "gpu.launch ";
  os << "blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)\n";
  indent();
  os << "          threads(%tx, %ty, %tz) in (%block_x = %c5, %block_y = %c1, %block_z = %c1) {\n";

  addIndent();
  indent();
  os << "// Kernel body\n";
  indent();
  os << "...\n";
  indent();
  os << "gpu.terminate\n";
  reduceIndent();
  indent();
  os << "}\n\n";
  indent();
  os << "return\n";
}

void ModuleEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = array.getType().cast<ShapedType>();
  if (arrayType.hasStaticShape()) {
    auto memref = array.getType().dyn_cast<MemRefType>();
    if (memref) {
      // auto attr = memref.getMemorySpace();
      // if (attr &&
      //     attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
      //   // Value has been declared before or is a constant number.
      //   if (isDeclared(array)) {
      //     os << getName(array);
      //     return;
      //   }
      //   // print stream type
      //   os << "hls::stream< " << getTypeName(array) << " > ";
      //   if (isFunc) {
      //     os << "&"; // pass by reference
      //   }
      //   // Add the new value to nameTable and emit its name.
      //   os << addName(array, /*isPtr=*/false, name);
      //   // Add original array declaration as comment
      //   os << " /* ";
      //   emitValue(array, 0, false, name);
      //   for (auto &shape : arrayType.getShape())
      //     os << "[" << shape << "]";
      //   os << " */";
      // }
      if (false) {
        // TODO
      } else {
        // emitValue(array, 0, false, name);
        os << "%" << addName(array, false) << ": " << memref;
        // os << ": " << memref;
        // if (arrayType.getShape().size() == 1 && arrayType.getShape()[0] == 1) {
        //   // do nothing;
        // } else {
        //   for (auto &shape : arrayType.getShape())
        //     os << "[" << shape << "]";
        // }
      }
    } 
    //else { // tensor
    //   emitValue(array, 0, false, name);
    // }
  } 
  // else
  //   emitValue(array, /*rank=*/0, /*isPtr=*/true, name);
}

void ModuleEmitter::emitFunction(FuncOp func) {
  // if (func->hasAttr("bit"))
  //   BIT_FLAG = true;

  // if (func.getBlocks().size() != 1)
  //   emitError(func, "has zero or more than one basic blocks.");

  // if (func->hasAttr("top"))
  //   os << "/// This is top function.\n";

  // Emit function signature.
  addIndent();
  indent();
  os << "func @" << func.getName() << " (";

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit function arguments.
  unsigned int argIdx = 0;
  // std::vector<std::string> input_args;
  // if (func->hasAttr("inputs")) {
  //   std::string input_names =
  //       func->getAttr("inputs").cast<StringAttr>().getValue().str();
  //   input_args = split_names(input_names);
  // }
  // std::string output_names;
  // if (func->hasAttr("outputs")) {
  //   output_names = func->getAttr("outputs").cast<StringAttr>().getValue().str();
  //   // suppose only one output
  //   input_args.push_back(output_names);
  // }

  for (auto &arg : func.getArguments()) {
    if (arg.getType().isa<ShapedType>()) {
      // TODO: for now, input_args.size() == 0
      // if (input_args.size() == 0) {
      emitArrayDecl(arg, true);
      // } else {
        // emitArrayDecl(arg, true, input_args[argIdx]);
      // }
    } else {
      os << "N/A\n";
    }

    if (argIdx++ != func.getNumArguments() - 1)
      os << ", ";

    portList.push_back(arg);
  }

  os << ") {";
  emitInfoAndNewLine(func);

  // Emit function body
  addIndent();
  emitBlock(func.front());
  reduceIndent();
  indent();
  os << "}\n";

  // End module
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string run_instr = R"XXX(// RUN: hcl-opt -opt %s | FileCheck %s)XXX";
  os << run_instr << "\n\n";
  std::string module_header = R"XXX(module {)XXX";
  os << module_header << "\n";
  for (auto op : module.getOps<FuncOp>()) {
    emitFunction(op);
  }
}

//===----------------------------------------------------------------------===//
// Entry of hcl-translate
//===----------------------------------------------------------------------===//

LogicalResult hcl::emitMlirGpu(ModuleOp module, llvm::raw_ostream &os) {
  HCLEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void hcl::registerEmitMlirGpuTranslation() {
  static TranslateFromMLIRRegistration toMlirGpu(
    "emit-gpu", emitMlirGpu, [&](DialectRegistry &registry) {
      // clang-format off
      registry.insert<
        mlir::hcl::HeteroCLDialect,
        mlir::StandardOpsDialect,
        mlir::arith::ArithmeticDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect,
        mlir::AffineDialect,
        mlir::math::MathDialect,
        mlir::memref::MemRefDialect
      >();
      // clang-format on
    });
}
