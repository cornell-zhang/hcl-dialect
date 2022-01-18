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

#include "hcl-c/Dialect/HCLTypes.h"
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
  static constexpr IsAFunctionTy isaFunction = hclMlirTypeIsALoopHandle;
  static constexpr const char *pyClassName = "LoopHandleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = hclMlirLoopHandleTypeGet(context->get());
          return PyLoopHandleType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a loop handle type.");
  }
};

/// StageHandleType.
class PyStageHandleType : public PyConcreteType<PyStageHandleType> {
public:
  static constexpr IsAFunctionTy isaFunction = hclMlirTypeIsAStageHandle;
  static constexpr const char *pyClassName = "StageHandleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = hclMlirStageHandleTypeGet(context->get());
          return PyStageHandleType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a stage handle type.");
  }
};

/// FixedType.
class PyFixedType : public PyConcreteType<PyFixedType> {
public:
  static constexpr IsAFunctionTy isaFunction = hclMlirTypeIsAFixedType;
  static constexpr const char *pyClassName = "FixedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](size_t width, size_t frac, DefaultingPyMlirContext context) {
          MlirType t = hclMlirFixedTypeGet(context->get(), width, frac);
          return PyFixedType(context->getRef(), t);
        },
        py::arg("width"), py::arg("frac"), py::arg("context") = py::none(), "Create a fixed type.");
  }
};

/// UFixedType.
class PyUFixedType : public PyConcreteType<PyUFixedType> {
public:
  static constexpr IsAFunctionTy isaFunction = hclMlirTypeIsAUFixedType;
  static constexpr const char *pyClassName = "UFixedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](size_t width, size_t frac, DefaultingPyMlirContext context) {
          MlirType t = hclMlirUFixedTypeGet(context->get(), width, frac);
          return PyUFixedType(context->getRef(), t);
        },
        py::arg("width"), py::arg("frac"), py::arg("context") = py::none(), "Create a ufixed type.");
  }
};

} // namespace

void mlir::python::populateHCLIRTypes(py::module &m) {
  PyLoopHandleType::bind(m);
  PyStageHandleType::bind(m);
  PyFixedType::bind(m);
  PyUFixedType::bind(m);
}