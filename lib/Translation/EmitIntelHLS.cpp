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
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public HCLEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(HCLEmitterState &state) : HCLEmitterBase(state) {}

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);
};
} // namespace

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for Intel High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>

#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel name in the global scope to reduce name mangling.
class Top;


int main() {

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

      // Submit a command group to the device queue.
      q.submit([&](handler& h) {

        // The SYCL runtime uses the accessors to infer data dependencies.
        // A "read" accessor must wait for data to be copied to the device
        // before the kernel can start. A "write no_init" accessor does not.
        accessor a(buf_a, h, read_only);
        accessor b(buf_b, h, read_only);
        accessor r(buf_r, h, write_only, no_init);

        // The kernel uses single_task rather than parallel_for.
        // The task's for loop is executed in pipeline parallel on the FPGA,
        // exploiting the same parallelism as an equivalent parallel_for.
        //
        //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
        h.single_task<Top>([=]() [[intel::kernel_args_restrict]] {
)XXX";

  // generate main body


  // generate epilogue
  std::string epilogue = R"XXX(
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

  // if (module.getName().hasValue() && module.getName().getValue() == "host") {
  os << header;
  os << epilogue;
  //   for (auto op : module.getOps<FuncOp>()) {
  //     if (op.getName() == "main")
  //       emitHostFunction(op);
  //     else
  //       emitFunction(op);
  //   }
  // } else {
  //   os << device_header;
  //   for (auto &op : *module.getBody()) {
  //     if (auto func = dyn_cast<FuncOp>(op))
  //       emitFunction(func);
  //     else
  //       emitError(&op, "is unsupported operation.");
  //   }
  // }
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