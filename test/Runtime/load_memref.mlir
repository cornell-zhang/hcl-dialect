// RUN: hcl-opt  %s 

module {
    llvm.mlir.global internal constant @str_global("test.txt")
    func.func private @loadMemrefI32(memref<*xi32>, !llvm.ptr<i8>)
    func.func @top () -> () {
        %0 = memref.alloc() : memref<1xi32>
        %1 = memref.cast %0 : memref<1xi32> to memref<*xi32>
        %2 = llvm.mlir.addressof @str_global : !llvm.ptr<array<8xi8>>
        %3 = llvm.mlir.constant(0 : index) : i64
        %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<8xi8>>, i64, i64) -> !llvm.ptr<i8>
        call @loadMemrefI32(%1, %4) : (memref<*xi32>, !llvm.ptr<i8>) -> ()
        return
    }
    func.func @main() -> () {
        call @top() : () -> ()
        return
    }
}