file(GLOB globbed *.cpp)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRHCLConversion
    ${globbed}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hcl

    DEPENDS
    MLIRHeteroCLOpsIncGen
    MLIRHeteroCLTypesIncGen
    MLIRHeteroCLPassesIncGen
    MLIRHeteroCLConversionPassesIncGen

    LINK_LIBS PUBLIC
    ${conversion_libs}
    MLIRIR
    MLIRPass
    MLIRHeteroCL
    MLIRHCLSupport
)

if (CUDA_BACKEND)
  target_compile_definitions(obj.MLIRHCLConversion PRIVATE CUDA_BACKEND_ENABLED=1)
  target_include_directories(MLIRHCLConversion PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CUDA enabled for MLIRHCLConversion")
endif()
