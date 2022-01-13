add_mlir_conversion_library(MLIRReconcileUnrealizedCasts
  ReconcileUnrealizedCasts.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Transforms/ReconcileUnrealizedCasts
  
  DEPENDS
  MLIRHeteroCLPassesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
)