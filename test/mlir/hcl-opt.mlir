// RUN: hcl-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK: hcl

%0 = constant 2 : i32
%1 = hcl.foo %0 : i32
