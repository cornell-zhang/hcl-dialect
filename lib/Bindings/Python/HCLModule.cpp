/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "hcl/Bindings/Python/HCLModule.h"
#include "hcl-c/Dialect/Dialects.h"
#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl-c/Dialect/Registration.h"
#include "hcl-c/Translation/EmitIntelHLS.h"
#include "hcl-c/Translation/EmitVivadoHLS.h"
#include "hcl/Conversion/Passes.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/Passes.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;
using namespace hcl;

//===----------------------------------------------------------------------===//
// Customized Python classes
//===----------------------------------------------------------------------===//

// PybindUtils.h
class PyFileAccumulator {
public:
  PyFileAccumulator(const pybind11::object &fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire acquire;
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

private:
  pybind11::object pyWriteFunction;
  bool binary;
};

//===----------------------------------------------------------------------===//
// Loop transform APIs
//===----------------------------------------------------------------------===//

static bool loopTransformation(MlirModule &mlir_mod) {
  py::gil_scoped_release();
  auto mod = unwrap(mlir_mod);
  return applyLoopTransformation(mod);
}

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitVivadoHls(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitVivadoHls(mod, accum.getCallback(), accum.getUserData()));
}

static bool emitIntelHls(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitIntelHls(mod, accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// Lowering APIs
//===----------------------------------------------------------------------===//

static bool lowerHCLToLLVM(MlirModule &mlir_mod, MlirContext &mlir_ctx) {
  auto mod = unwrap(mlir_mod);
  auto ctx = unwrap(mlir_ctx);
  return applyHCLToLLVMLoweringPass(mod, *ctx);
}

static bool lowerFixedPointToInteger(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyFixedPointToInteger(mod);
}

static bool lowerAnyWidthInteger(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyAnyWidthInteger(mod);
}

static bool moveReturnToInput(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyMoveReturnToInput(mod);
}

static bool lowerCompositeType(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerCompositeType(mod);
}

static bool lowerBitOps(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerBitOps(mod);
}

static bool legalizeCast(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLegalizeCast(mod);
}

static bool removeStrideMap(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyRemoveStrideMap(mod);
}

static bool lowerPrintOps(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerPrintOps(mod);
}

//===----------------------------------------------------------------------===//
// Utility pass APIs
//===----------------------------------------------------------------------===//
static bool memRefDCE(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyMemRefDCE(mod);
}

//===----------------------------------------------------------------------===//
// HCL Python module definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_hcl, m) {
  m.doc() = "HCL Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  // register passes
  hclMlirRegisterAllPasses();

  auto hcl_m = m.def_submodule("hcl");

  // register dialects
  hcl_m.def(
      "register_dialect",
      [](MlirContext context) {
        MlirDialectHandle hcl = mlirGetDialectHandle__hcl__();
        mlirDialectHandleRegisterDialect(hcl, context);
        mlirDialectHandleLoadDialect(hcl, context);
      },
      py::arg("context") = py::none());

  // Declare customized types and attributes
  populateHCLIRTypes(hcl_m);
  populateHCLAttributes(hcl_m);

  // Loop transform APIs.
  hcl_m.def("loop_transformation", &loopTransformation);

  // Codegen APIs.
  hcl_m.def("emit_vhls", &emitVivadoHls);
  hcl_m.def("emit_ihls", &emitIntelHls);

  // LLVM backend APIs.
  hcl_m.def("lower_hcl_to_llvm", &lowerHCLToLLVM);
  hcl_m.def("lower_fixed_to_int", &lowerFixedPointToInteger);
  hcl_m.def("lower_anywidth_int", &lowerAnyWidthInteger);
  hcl_m.def("move_return_to_input", &moveReturnToInput);

  // Lowering APIs.
  hcl_m.def("lower_composite_type", &lowerCompositeType);
  hcl_m.def("lower_bit_ops", &lowerBitOps);
  hcl_m.def("legalize_cast", &legalizeCast);
  hcl_m.def("remove_stride_map", &removeStrideMap);
  hcl_m.def("lower_print_ops", &lowerPrintOps);

  // Utility pass APIs.
  hcl_m.def("memref_dce", &memRefDCE);
}
