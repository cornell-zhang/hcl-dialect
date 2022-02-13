//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "hcl/Bindings/Python/HCLModule.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "hcl-c/Dialect/Dialects.h"
#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl-c/Dialect/Registration.h"
#include "hcl-c/Translation/EmitHLSCpp.h"
#include "hcl/Conversion/HCLToLLVM.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/Passes.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/Standard.h"
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

// static bool hostXcelSeparation(PyModule &pyhost, PyModule &pyxcel,
//                                PyModule &pyextern, py::dict pydevice_map,
//                                py::list pygraph_roots, py::dict subgraph) {
//   py::gil_scoped_release();
//   auto host_mod = unwrap(pyhost.get());
//   auto xcel_mod = unwrap(pyxcel.get());
//   auto extern_mod = unwrap(pyextern.get());
//   std::map<std::string, std::string> device_map;
//   for (auto item : pydevice_map) {
//     device_map[item.first.cast<std::string>()] =
//         item.second.cast<std::string>();
//   }
//   std::vector<std::string> graph_roots;
//   for (auto root : pygraph_roots) {
//     graph_roots.push_back(root.cast<std::string>());
//   }
//   std::vector<std::string> inputs;
//   for (auto input : subgraph["inputs"]) {
//     inputs.push_back(input.cast<std::string>());
//   }
//   std::vector<std::string> outputs;
//   for (auto output : subgraph["outputs"]) {
//     outputs.push_back(output.cast<std::string>());
//   }
//   return applyHostXcelSeparation(host_mod, xcel_mod, extern_mod, device_map,
//                                  graph_roots, inputs, outputs);
// }

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitHlsCpp(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitHlsCpp(mod, accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// Lowering APIs
//===----------------------------------------------------------------------===//

static bool lowerHCLToLLVM(MlirModule &mlir_mod, MlirContext &mlir_ctx) {
  auto mod = unwrap(mlir_mod);
  auto ctx = unwrap(mlir_ctx);
  return applyHCLToLLVMLoweringPass(mod, *ctx);
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
  // m.def("host_device_separation", &hostXcelSeparation);

  // Codegen APIs.
  hcl_m.def("emit_hlscpp", &emitHlsCpp);
  hcl_m.def("lower_hcl_to_llvm", &lowerHCLToLLVM);
}
