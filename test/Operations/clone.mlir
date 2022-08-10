// RUN: hcl-opt %s
module {
  func.func @sub_func () -> () {
    return
  }
  func.func @top() -> (){
    call @sub_func() : () -> ()
    %s = hcl.create_op_handle "sub_func"
    hcl.clone(@sub_func, %s)
    return
  }
}