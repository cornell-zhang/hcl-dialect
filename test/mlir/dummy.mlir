// RUN: hcl-opt %s | hcl-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = hcl.foo %{{.*}} : i32
        %res = hcl.foo %0 : i32
        return
    }
}
