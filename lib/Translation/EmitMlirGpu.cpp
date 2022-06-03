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
  void emitFunction(FuncOp func);
};
} // namespace

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
  os << "func @" << func.getName() << " (\n";

  // Emit function arguments
  unsigned int argIdx = 0;
  SmallVector<Value, 8> portList;
  addIndent();
  for (auto &arg : func.getArguments()) {
    indent();
    if (arg.getType().isa<ShapedType>()) {
      os << arg;
    } else {
      os << "NO\n";
    }

    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";

    portList.push_back(arg);
  }
  os << "\n";
  reduceIndent();

  // Emit function body
  indent();
  os << ") {\n";
  // emitInfoAndNewLine(func);

  // Data register
  // TODO: do we really need to cast the memref?
  addIndent();
  // indent();
  // os << "%cast_src0 = memref.cast %src0 : memref<5xi32> to memref<*xi32>\n";
  indent();
  os << "gpu.host_register %cast_A : memref<256xf32>\n";
  // indent();
  // os << "%cast_src1 = memref.cast %src1 : memref<5xi32> to memref<*xi32>\n";
  indent();
  os << "gpu.host_register %cast_B : memref<256xf32>\n";
  // indent();
  // os << "%cast_dest = memref.cast %dest : memref<5xi32> to memref<*xi32>\n";
  indent();
  os << "gpu.host_register %cast_C : memref<256xf32>\n";

  // GPU Kernel definition
  os << "\n";
  indent();
  os << "gpu.launch ";

  // TODO: Calculate dimensions based off of split?

  // Block dimension definition
  os << "blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)\n";

  // Thread dimension definition
  indent();
  os << "          threads(%tx, %ty, %tz) in (%block_x = %c5, %block_y = %c1, %block_z = %c1) {\n";

  // Kernel body
  addIndent();

  indent();
  os << "%a = memref.load %cast_A[%tx] : memref<256xf32>\n";
  indent();
  os << "%b = memref.load %cast_B[%tx] : memref<256xf32>\n";
  indent();
  os << "%sum = arith.addf %a, %b : f32\n";
  indent();
  os << "memref.store %sum, %cast_C[%tx] : memref<256xf32>\n";

  indent();
  os << "gpu.terminator\n";

  reduceIndent();
  indent();
  os << "}\n";

  // End function
  indent();
  os << "return\n";

  reduceIndent();
  indent();
  os << "}\n";

  // End module
  os << "}";

  // An empty line.
  os << "\n";
}

void ModuleEmitter::emitModule(ModuleOp module) {
  std::string run_instr = R"XXX(// RUN: hcl-opt -opt %s | FileCheck %s)XXX";
  os << run_instr << "\n\n";
  std::string module_header = R"XXX(module {)XXX";
  os << module_header << "\n";
  for (auto op : module.getOps<FuncOp>()) {
    emitFunction(op);
  }
}

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
