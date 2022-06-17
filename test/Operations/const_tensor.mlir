// RUN: hcl-opt --fixed-to-integer --jit %s
module {
  memref.global "private" @gv_cst : memref<2x2xi64> = dense<[[8, 0], [10, 20]]>

  func @top() -> () {
    %0 = hcl.get_global_fixed @gv_cst : memref<2x2x!hcl.Fixed<32,2>>
    hcl.print (%0) {format = "%.0f \n"} : memref<2x2x!hcl.Fixed<32,2>> 
    return
  }
}

