//===- HeteroCLTypes.td - hcl dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_ATTRS
#define HCL_ATTRS

include "HeteroCLDialect.td"

def Loop : HeteroCL_Attr<"Loop"> {
  let mnemonic = "Loop";
  let parameters = (ins "hcl::LoopHandleType":$loopName);
  let printer = [{
    StringRef loopName = getImpl()->loopName.getLoopName();
    $_printer << "Loop <" << loopName.str() << ">";
  }];
  // llvm::printEscapedString(getImpl()->loopName, $_printer.getStream());
  let parser = [{
    if (parser.parseLess())
      return Attribute();
    StringRef loopName;
    if ($_parser.parseOptionalString(&loopName))
      return Attribute();
    if ($_parser.parseGreater())
      return Attribute();
    return get($_ctxt, hcl::LoopHandleType::get($_ctxt, loopName));
  }];
}

#endif // HCL_ATTRS