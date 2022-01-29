//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "hcl/Bindings/Python/HCLModule.h"
#include "IRModule.h"
#include "hcl-c/Dialect/Dialects.h"
#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl-c/Dialect/Registration.h"
#include "hcl-c/Translation/EmitHLSCpp.h"
#include "hcl/Conversion/HCLToLLVM.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/HostXcelSeparation.h"
#include "hcl/Transforms/LoopTransformations.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/CAPI/IR.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

namespace py = pybind11;

using namespace mlir;
using namespace mlir::python;
using namespace hcl;

//===----------------------------------------------------------------------===//
// Customized Python classes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Loop transform APIs
//===----------------------------------------------------------------------===//

static bool loopTransformation(PyModule &pymod) {
  py::gil_scoped_release();
  auto mod = unwrap(pymod.get());
  return applyLoopTransformation(mod);
}

static bool hostXcelSeparation(PyModule &host, PyModule &device,
                               py::dict pydevice_map, py::list pygraph_roots,
                               py::dict subgraph) {
  py::gil_scoped_release();
  auto host_mod = unwrap(host.get());
  auto device_mod = unwrap(device.get());
  std::map<std::string, std::string> device_map;
  for (auto item : pydevice_map) {
    device_map[item.first.cast<std::string>()] =
        item.second.cast<std::string>();
  }
  std::vector<std::string> graph_roots;
  for (auto root : pygraph_roots) {
    graph_roots.push_back(root.cast<std::string>());
  }
  std::vector<std::string> inputs;
  for (auto input : subgraph["inputs"]) {
    inputs.push_back(input.cast<std::string>());
  }
  std::vector<std::string> outputs;
  for (auto output : subgraph["outputs"]) {
    outputs.push_back(output.cast<std::string>());
  }
  return applyHostXcelSeparation(host_mod, device_mod, device_map, graph_roots,
                                 inputs, outputs);
}

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitHlsCpp(PyModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitHlsCpp(mod.get(), accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// Lowering APIs
//===----------------------------------------------------------------------===//

static bool lowerHCLToLLVM(PyModule &module, PyMlirContext &context) {
  auto mod = unwrap(module.get());
  auto ctx = unwrap(context.get());
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

  m.def("register_dialects", [](py::object capsule) {
    // Get the MlirContext capsule from PyMlirContext capsule.
    auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
    MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

    MlirDialectHandle hcl = mlirGetDialectHandle__hcl__();
    mlirDialectHandleRegisterDialect(hcl, context);
    mlirDialectHandleLoadDialect(hcl, context);
  });

  // Type construction APIs.
  populateHCLIRTypes(m);
  populateHCLAttributes(m);

  // Loop transform APIs.
  m.def("loop_transformation", &loopTransformation);
  m.def("host_device_separation", &hostXcelSeparation);

  // Emission APIs.
  m.def("emit_hlscpp", &emitHlsCpp);

  m.def("lower_hcl_to_llvm", &lowerHCLToLLVM);
}
