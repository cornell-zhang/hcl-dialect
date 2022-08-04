//===- hcl-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitIntelHLS.h"
#include "hcl/Translation/EmitVivadoHLS.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#ifdef OPENSCOP
#include "hcl/Target/OpenSCoP/ExtractScopStmt.h"
#endif

#include "hcl/Dialect/HeteroCLDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::hcl::registerEmitVivadoHLSTranslation();
  mlir::hcl::registerEmitIntelHLSTranslation();
#ifdef OPENSCOP
  mlir::hcl::registerToOpenScopExtractTranslation();
#endif

  return failed(mlir::mlirTranslateMain(
      argc, argv, "HeteroCL MLIR Dialect Translation Tool"));
}
