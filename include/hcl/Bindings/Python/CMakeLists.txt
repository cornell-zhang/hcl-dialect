################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(MLIRPythonSources)
declare_mlir_python_sources(MLIRPythonSources.Dialects
  ADD_TO_PARENT MLIRPythonSources)

################################################################################
# Declare native Python extension
################################################################################

set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")
set(HCL_BINARY_DIR "${PROJECT_BINARY_DIR}/tools/hcl")

# declare_mlir_python_sources(HCLBindingsPythonExtension)
# 
# declare_mlir_python_extension(HCLBindingsPythonExtension.Core
#   MODULE_NAME _hcl
#   ADD_TO_PARENT HCLBindingsPythonExtension
#   SOURCES
#     HCLModule.cpp
#     ${MLIR_PYTHON_SOURCE_DIR}/PybindUtils.cpp
#   EMBED_CAPI_LINK_LIBS
#     MLIRHCLCAPIHLSCpp
#   PRIVATE_LINK_LIBS
#     LLVMSupport
#     MLIRHCLConversion
#     MLIRHCLTransforms
# )

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(HCLBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    _ods_common.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/lib/Bindings/Python/hcl/dialects)

declare_mlir_python_sources(HCLBindingsPythonSources.Dialects
  ADD_TO_PARENT HCLBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HCLBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE HeteroCLBinding.td
  SOURCES
    hcl.py
  DIALECT_NAME hcl)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(HCLBindingsPythonCAPI
  INSTALL_COMPONENT HCLBindingsPythonModules
  INSTALL_DESTINATION python_packages/hcl_core/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under hcl, this would change to
  # .../hcl/_mlir_libs
  OUTPUT_DIRECTORY "${HCL_BINARY_DIR}/python_packages/hcl_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    HCLBindingsPythonSources
    #HCLBindingsPythonExtension
)

# Bundle the MLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the hcl/
# folder and be accessible under the unified "import hcl..." namespace.
add_mlir_python_modules(HCLMLIRPythonModules
  ROOT_PREFIX "${HCL_BINARY_DIR}/python_packages/hcl_core/mlir"
  INSTALL_PREFIX "python_packages/hcl_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    # We need the hcl extensions co-located with the MLIR extensions. When
    # the namespace is unified, this moves to the below.
    #HCLBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    HCLBindingsPythonCAPI
)

# Bundle the HCL python sources into our package.
add_mlir_python_modules(HCLPythonModules
  ROOT_PREFIX "${HCL_BINARY_DIR}/python_packages/hcl_core/hcl"
  INSTALL_PREFIX "python_packages/hcl_core/hcl"
  DECLARED_SOURCES
    HCLBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    HCLBindingsPythonCAPI
)
