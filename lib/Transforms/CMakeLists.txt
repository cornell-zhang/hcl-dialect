add_mlir_library(MLIRHCLPasses
    LoopTransformations.cpp
    HostXcelSeparation.cpp
    FixedPointToInteger.cpp
    AnyWidthInteger.cpp
    FloatingPointToInteger.cpp
    MoveReturnToInput.cpp

    Passes.cpp

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