file(GLOB globbed *.cpp)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRHCLPasses
    ${globbed}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hcl

    DEPENDS
    MLIRHeteroCLOpsIncGen
    MLIRHeteroCLTypesIncGen

    LINK_LIBS PUBLIC
    ${conversion_libs}
    MLIRIR
    MLIRPass
    MLIRHeteroCL
    MLIRHCLSupport
)