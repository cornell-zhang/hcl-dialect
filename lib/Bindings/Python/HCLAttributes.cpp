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

#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl/Bindings/Python/HCLModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace mlir;
using namespace mlir::python;

namespace {

class PyIntegerSetAttribute
    : public PyConcreteAttribute<PyIntegerSetAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAIntegerSet;
  static constexpr const char *pyClassName = "IntegerSetAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyIntegerSet &IntegerSet) {
          MlirAttribute attr = mlirIntegerSetAttrGet(IntegerSet.get());
          return PyIntegerSetAttribute(IntegerSet.getContext(), attr);
        },
        py::arg("integer_set"), "Gets an attribute wrapping an IntegerSet.");
  }
};

} // namespace

void mlir::python::populateHCLAttributes(py::module &m) {
  PyIntegerSetAttribute::bind(m);
}