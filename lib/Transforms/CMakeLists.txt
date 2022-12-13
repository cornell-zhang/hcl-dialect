add_mlir_library(MLIRHCLPasses
    LoopTransformations.cpp
    AnyWidthInteger.cpp
    MoveReturnToInput.cpp
    Passes.cpp
    LegalizeCast.cpp
    RemoveStrideMap.cpp
    MemRefDCE.cpp
    DataPlacement.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hcl

    DEPENDS
    MLIRHeteroCLOpsIncGen
    MLIRHeteroCLTypesIncGen
    MLIRHeteroCLPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRHeteroCL
    MLIRHCLSupport
)