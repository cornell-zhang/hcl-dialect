add_mlir_dialect_library(MLIRHeteroCL
        HeteroCLDialect.cpp
        HeteroCLOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/hcl

        DEPENDS
        MLIRHeteroCLOpsIncGen
        MLIRHeteroCLTypesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
