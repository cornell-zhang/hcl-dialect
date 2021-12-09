# HeteroCL Dialect Prototype

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with an  `opt`-like tool to operate on that dialect.

## Building

### Preliminary tools
- gcc >= 5.4
- cmake >= 3.13.4
- [ninja](https://ninja-build.org)

### Install LLVM 13.0.0
- Download LLVM from here: [llvm-project](https://github.com/llvm/llvm-project/releases/tag/llvmorg-13.0.0).
- Build:
```sh
cmake \
-G "Unix Makefiles" ../llvm \
-DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
-DCMAKE_INSTALL_PREFIX=/your/path/to/install/ \
-DBUILD_SHARED_LIBS=On \
-DLLVM_BUILD_LLVM_DYLIB=On \
-DLLVM_TARGETS_TO_BUILD="X86" \
-DCMAKE_BUILD_TYPE=RelWithDebInfo;
make -j20
```
### Build Our Dialect
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target hcl-opt
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

### For Internal Developers
LLVM doesn’t support newer GLIBCXX library thread API. We must build cmake and llvm with GLIBCXX lower than 3.4.20. I don’t think this is documented in LLVM’s guide. 

Error message: Undefined reference in standard C++ library
`libLLVMSupport.so: undefined reference to std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)())@GLIBCXX_3.4.22`

I recommend building on `zhang-x1` server, where GLIBCXX library is older.

Details about this issue: [group wiki](https://zhang-21.ece.cornell.edu/doku.php?id=research:personal:niansongzhang:tools:dylib)

### Run Our Dialect
```sh
./build/bin/hcl-opt ./test/mlir/matmul.mlir
```