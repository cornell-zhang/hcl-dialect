// RUN: hcl-opt %s 
module {
  func @top (%arg0 : !hcl.struct<i3, i3>) -> () {
    return
  }
} 
