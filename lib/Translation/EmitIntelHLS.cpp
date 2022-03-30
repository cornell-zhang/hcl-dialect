//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitIntelHLS.h"
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
// Utils
//===----------------------------------------------------------------------===//

static SmallString<16> getTypeName(Value val) {
  // Handle memref, tensor, and vector types.
  bool BIT_FLAG = false;
  auto valType = val.getType();
  if (auto arrayType = val.getType().dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle float types.
  if (valType.isa<Float32Type>())
    return SmallString<16>("float");
  else if (valType.isa<Float64Type>())
    return SmallString<16>("double");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      if (!BIT_FLAG)
        return SmallString<16>("bool");
      else
        return SmallString<16>("ac_uint<1>");
    } else {
      std::string signedness = "";
      if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
        signedness = "u";
      if (!BIT_FLAG) {
        switch (intType.getWidth()) {
        case 8:
        case 16:
        case 32:
        case 64:
          return SmallString<16>(signedness + "int" +
                                 std::to_string(intType.getWidth()) + "_t");
        default:
          return SmallString<16>("ac_" + signedness + "int<" +
                                 std::to_string(intType.getWidth()) + ">");
        }
      } else {
        return SmallString<16>("ac_" + signedness + "int<" +
                               std::to_string(intType.getWidth()) + ">");
      }
    }
  }

  // Handle (custom) fixed point types.
  else if (auto fixedType = valType.dyn_cast<hcl::FixedType>())
    return SmallString<16>(
        "ac_fixed<" + std::to_string(fixedType.getWidth()) + ", " +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ">");

  else if (auto ufixedType = valType.dyn_cast<hcl::UFixedType>())
    return SmallString<16>(
        "ac_ufixed<" + std::to_string(ufixedType.getWidth()) + ", " +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) + ">");
  else
    val.getDefiningOp()->emitError("has unsupported type.");

  return SmallString<16>();
}

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
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  void emitArrayDecl(Value array, bool isAccessor = false,
                     bool isReadOnly = false, std::string name = "");
  void emitFunction(FuncOp func, bool isAccessor = false);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
};
} // namespace

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              std::string name) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getTypeName(val) << " ";

  if (name == "") {
    // Add the new value to nameTable and emit its name.
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

void ModuleEmitter::emitArrayDecl(Value array, bool isAccessor, bool isReadOnly,
                                  std::string name) {
  auto arrayType = array.getType().cast<ShapedType>();
  assert(arrayType.hasStaticShape());
  auto memref = array.getType().dyn_cast<MemRefType>();
  assert(memref);
  if (!isAccessor) {
    os << "buffer<";
    os << getTypeName(array) << ", ";
    os << arrayType.getRank() << "> ";
    os << "buf_";
    emitValue(array, 0, false, name);
    os << "(";
    for (unsigned i = 0; i < arrayType.getRank(); ++i) {
      os << arrayType.getShape()[i];
      if (i != arrayType.getRank() - 1)
        os << ", ";
    }
    os << ");\n";
  } else {
    os << "accessor ";
    emitValue(array, 0, false, name);
    os << "(buf_";
    emitValue(array, 0, false, name);
    os << ", h";
    if (isReadOnly)
      os << ", read_only";
    os << ");\n";
  }
}

/// MLIR component and HLS C++ pragma emitters.
void ModuleEmitter::emitBlock(Block &block) {
  // for (auto &op : block) {
  //   if (ExprVisitor(*this).dispatchVisitor(&op))
  //     continue;

  //   if (StmtVisitor(*this).dispatchVisitor(&op))
  //     continue;

  //   emitError(&op, "can't be correctly emitted.");
  // }
}

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();
  os << "\n";
}

void ModuleEmitter::emitFunction(FuncOp func, bool isAccessor) {

  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  std::vector<std::string> input_args;
  if (func->hasAttr("inputs")) {
    std::string input_names =
        func->getAttr("inputs").cast<StringAttr>().getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names = func->getAttr("outputs").cast<StringAttr>().getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string extra_itypes = "";
  if (func->hasAttr("extra_itypes"))
    extra_itypes =
        func->getAttr("extra_itypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      extra_itypes += "x";
  }
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, extra_itypes[argIdx] == 'u');
    if (arg.getType().isa<ShapedType>()) {
      if (input_args.size() == 0) {
        emitArrayDecl(arg, isAccessor, true);
      } else {
        emitArrayDecl(arg, isAccessor, true, input_args[argIdx]);
      }
    } else {
      if (input_args.size() == 0) {
        emitValue(arg);
      } else {
        emitValue(arg, 0, false, input_args[argIdx]);
      }
    }

    portList.push_back(arg);
    argIdx++;
  }

  // Emit results.
  auto args = func.getArguments();
  std::string extra_otypes = "";
  if (func->hasAttr("extra_otypes"))
    extra_otypes =
        func->getAttr("extra_otypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      extra_otypes += "x";
  }
  if (auto funcReturn = dyn_cast<ReturnOp>(func.front().getTerminator())) {
    unsigned idx = 0;
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        indent();

        // TODO: a known bug, cannot return a value twice, e.g. return %0, %0
        // : index, index. However, typically this should not happen.
        fixUnsignedType(result, extra_otypes[idx] == 'u');
        if (result.getType().isa<ShapedType>()) {
          if (output_names != "")
            emitArrayDecl(result, isAccessor, false);
          else
            emitArrayDecl(result, isAccessor, false, output_names);
        } else {
          // In Vivado HLS, pointer indicates the value is an output.
          if (output_names != "")
            emitValue(result, /*rank=*/0, /*isPtr=*/true);
          else
            emitValue(result, /*rank=*/0, /*isPtr=*/true, output_names);
        }

        portList.push_back(result);
      }
      idx += 1;
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  emitInfoAndNewLine(func);
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string snippet = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for Intel High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel name in the global scope to reduce name mangling.
// This is an FPGA best practice that makes it easier to identify the kernel in 
// the optimization reports.
class Top;


int main() {
)XXX";
  os << snippet;

  snippet = R"XXX(
  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  try {

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, dpc_common::exception_handler);

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    {
      // Create buffers to share data between host and device.
      // The runtime will copy the necessary data to the FPGA device memory
      // when the kernel is launched.
)XXX";
  os << snippet;

  addIndent();
  addIndent();
  addIndent();
  // generate initial buffers (function arguments)
  // TODO: can only support one function now!
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<FuncOp>(op))
      emitFunction(func);
    else
      emitError(&op, "is unsupported operation.");
  }

  snippet = R"XXX(
      // Submit a command group to the device queue.
      q.submit([&](handler& h) {

        // The SYCL runtime uses the accessors to infer data dependencies.
        // A "read" accessor must wait for data to be copied to the device
        // before the kernel can start. A "write no_init" accessor does not.
)XXX";
  os << snippet;

  addIndent();
  addIndent();
  // generate accessors
  // TODO: can only support one function now!
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<FuncOp>(op))
      emitFunction(func, true);
    else
      emitError(&op, "is unsupported operation.");
  }

  snippet = R"XXX(

        // The kernel uses single_task rather than parallel_for.
        // The task's for loop is executed in pipeline parallel on the FPGA,
        // exploiting the same parallelism as an equivalent parallel_for.
        //
        //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
        h.single_task<Top>([=]() [[intel::kernel_args_restrict]] {
)XXX";
  os << snippet;

  // Emit function body.
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<FuncOp>(op)) {
      addIndent();
      emitBlock(func.front());
      reduceIndent();
    }
  }

  snippet = R"XXX(
        });
      });

      // The buffer destructor is invoked when the buffers pass out of scope.
      // buf_r's destructor updates the content of vec_r on the host.
    }

    // The queue destructor is invoked when q passes out of scope.
    // q's destructor invokes q's exception handler on any device exceptions.
  }
  catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  return 0;
}
)XXX";
  os << snippet;
}

//===----------------------------------------------------------------------===//
// Entry of hcl-translate
//===----------------------------------------------------------------------===//

LogicalResult hcl::emitIntelHLS(ModuleOp module, llvm::raw_ostream &os) {
  HCLEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void hcl::registerEmitIntelHLSTranslation() {
  static TranslateFromMLIRRegistration toIntelHLS(
      "emit-intel-hls", emitIntelHLS, [&](DialectRegistry &registry) {
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