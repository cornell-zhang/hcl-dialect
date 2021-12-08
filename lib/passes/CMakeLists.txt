file(GLOB globbed *.cpp)

add_mlir_dialect_library(MLIRHCLPasses
    ${globbed}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hcl

    DEPENDS
    MLIRHeteroCLOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)