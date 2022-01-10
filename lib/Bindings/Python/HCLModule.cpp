//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "hcl/Bindings/Python/PybindAdaptors.h"

#include "hcl-c/EmitHLSCpp.h"
#include "hcl-c/HCL.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/HeteroCLPasses.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/../../lib/Bindings/Python/IRModule.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/CAPI/IR.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Signals.h"

#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>

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

static bool loopTransformation(MlirOperation op) {
  py::gil_scoped_release();
  auto func = dyn_cast<FuncOp>(unwrap(op));
  if (!func)
    throw SetPyError(PyExc_ValueError, "targeted operation not a function");
  return applyLoopTransformation(func);
}

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitHlsCpp(MlirModule mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitHlsCpp(mod, accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// HCL Python module definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_hcl, m) {
  m.doc() = "HCL Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  m.def("register_dialects", [](py::object capsule) {
    // Get the MlirContext capsule from PyMlirContext capsule.
    auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
    MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

    MlirDialectHandle hcl = mlirGetDialectHandle__hcl__();
    mlirDialectHandleRegisterDialect(hcl, context);
    mlirDialectHandleLoadDialect(hcl, context);
  });

  // Loop transform APIs.
  m.def("loop_transformation", &loopTransformation);

  // Emission APIs.
  m.def("emit_hlscpp", &emitHlsCpp);
}
