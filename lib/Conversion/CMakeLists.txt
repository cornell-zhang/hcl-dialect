if (CUDA_BACKEND)
  file(GLOB globbed *.cpp EXCEPT NVVMToCubin.cpp)
else()
  file(GLOB globbed *.cpp)
endif()

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

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

if (CUDA_BACKEND)
  target_compile_definitions(MLIRHCLConversion PRIVATE CUDA_ENABLED)
  target_include_directories(MLIRHCLConversion PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()