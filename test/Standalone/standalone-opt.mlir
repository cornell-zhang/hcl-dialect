// RUN: standalone-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK: standalone

%0 = constant 2 : i32
%1 = standalone.foo %0 : i32
