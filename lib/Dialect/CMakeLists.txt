add_mlir_dialect_library(MLIRHeteroCL
        HeteroCLDialect.cpp
        HeteroCLOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/hcl

        DEPENDS
        MLIRHeteroCLOpsIncGen
        MLIRHeteroCLTypesIncGen
        MLIRHeteroCLAttrsIncGen
        MLIRHeteroCLEnumsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)

add_subdirectory(TransformOps)
