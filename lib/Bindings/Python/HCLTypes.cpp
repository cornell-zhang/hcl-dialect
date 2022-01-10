//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"
#include "PybindUtils.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"

#include "hcl-c/HCL.h"
#include "hcl/Bindings/Python/HCLModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace mlir;
using namespace mlir::python;

namespace {

/// LoopHandleType.
class PyLoopHandleType : public PyConcreteType<PyLoopHandleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsALoopHandle;
  static constexpr const char *pyClassName = "LoopHandleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirLoopHandleTypeGet(context->get());
          return PyLoopHandleType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a loop handle type.");
  }
};

/// StageHandleType.
class PyStageHandleType : public PyConcreteType<PyStageHandleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAStageHandle;
  static constexpr const char *pyClassName = "StageHandleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirStageHandleTypeGet(context->get());
          return PyStageHandleType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a stage handle type.");
  }
};

} // namespace

void mlir::python::populateHCLIRTypes(py::module &m) {
  PyLoopHandleType::bind(m);
  PyStageHandleType::bind(m);
}