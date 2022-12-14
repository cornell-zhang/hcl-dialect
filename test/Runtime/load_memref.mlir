// RUN: hcl-opt  %s

module {
    func.func private @loadMemrefI32(memref<*xi32>)
    func.func @top () -> () {
        %0 = memref.alloc() : memref<1xi32>
        %1 = memref.cast %0 : memref<1xi32> to memref<*xi32>
        call @loadMemrefI32(%1) : (memref<*xi32>) -> ()
        return
    }
}