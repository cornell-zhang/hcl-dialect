include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(HCL_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HCL_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/hcl/python_packages")
set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")
set(HCL_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python")

include_directories(${MLIR_PYTHON_SOURCE_DIR})

# We vendor our own MLIR instance in the `hcl_mlir` namespace.
#add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=hcl_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(HCLMLIRPythonSources)
declare_mlir_python_sources(HCLMLIRPythonExtensions)

declare_mlir_python_sources(HCLMLIRPythonSources.Dialects
  ROOT_DIR "${HCL_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT HCLMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HCLMLIRPythonSources.Dialects
  ROOT_DIR "${HCL_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE HeteroCLBinding.td
  SOURCES
    __init__.py
    _ods_common.py
    build_ir.py
  DIALECT_NAME hcl
)

# Only a temporal solution for Affine dialect Python binding
# Ref: https://github.com/llvm/llvm-project/commit/8c1b785ce110b754c2112906021a929ddd32f587
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HCLMLIRPythonSources.Dialects
  ROOT_DIR "${HCL_MLIR_PYTHON_ROOT_DIR}"
  #TD_FILE affine/AffineOps.td
  SOURCES
    affine/__init__.py
    affine/_ods_common.py
    affine/_affine_ops_ext.py
    affine/_affine_ops_gen.py
  DIALECT_NAME affine
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(HCLMLIRPythonExtensions.Main
  MODULE_NAME _hcl
  ADD_TO_PARENT HCLMLIRPythonExtensions
  SOURCES
    ${HCL_PYTHON_SOURCE_DIR}/HCLModule.cpp
    ${HCL_PYTHON_SOURCE_DIR}/HCLTypes.cpp
    ${MLIR_PYTHON_SOURCE_DIR}/IRCore.cpp
    ${MLIR_PYTHON_SOURCE_DIR}/IRModule.cpp
    ${MLIR_PYTHON_SOURCE_DIR}/PybindUtils.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRHCLCAPIHLSCpp
    MLIRHCLPasses
    MLIRHCLCAPIEmitHLSCpp
  PRIVATE_LINK_LIBS
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIRegistration
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  HCLMLIRPythonSources
  HCLMLIRPythonExtensions
)

add_mlir_python_common_capi_library(HCLMLIRAggregateCAPI
  INSTALL_COMPONENT HCLMLIRPythonModules
  INSTALL_DESTINATION python_packages/hcl_core/hcl_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${HCL_MLIR_PYTHON_PACKAGES_DIR}/hcl_core/hcl_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(HCLMLIRPythonModules
  ROOT_PREFIX "${HCL_MLIR_PYTHON_PACKAGES_DIR}/hcl_core/hcl_mlir"
  INSTALL_PREFIX "python_packages/hcl_core/hcl_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    HCLMLIRAggregateCAPI
  )