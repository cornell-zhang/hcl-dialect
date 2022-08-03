file(GLOB globbed *.cpp)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

#TODO(Niansong): Can we split this into two parts?
add_mlir_dialect_library(MLIRHCLConversion
    ${globbed}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hcl

    DEPENDS
    MLIRHeteroCLOpsIncGen
    MLIRHeteroCLTypesIncGen
    MLIRHeteroCLPassesIncGen

    LINK_LIBS PUBLIC
    ${conversion_libs}
    MLIRIR
    MLIRPass
    MLIRHeteroCL
    MLIRHCLSupport
)

# Configure CUDA language support. Using check_language first allows us to
# give a custom error message.
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  target_compile_definitions(MLIRHCLConversion PRIVATE HCL_GPU_ENABLED=1)
  target_include_directories(MLIRHCLConversion PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  message(SEND_WARNING "MLIRHCLConversion is not configured to support CUDA.")
  target_compile_definitions(MLIRHCLConversion PRIVATE HCL_GPU_ENABLED=0)
endif()