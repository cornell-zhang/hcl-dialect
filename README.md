# HeteroCL Dialect Prototype

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with an  `opt`-like tool to operate on that dialect.

## Building

### Preliminary tools
- gcc >= 5.4
- cmake >= 3.13.4

### Install LLVM 13.0.0
- Download LLVM from [llvm-project](https://github.com/llvm/llvm-project/releases/tag/llvmorg-13.0.0) or checkout the Github branch
```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout tags/llvmorg-13.0.0 -b v13.0.0
```

- Build:
```sh
mkdir build && cd build
cmake -G "Unix Makefiles" ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
make -j
```

### Build HeteroCL Dialect
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
git clone https://github.com/zzzDavid/hcl-dialect-prototype.git
cd hcl-dialect-prototype
mkdir build && cd build
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
make -j
```


## Run HeteroCL Dialect
```sh
./bin/hcl-opt ../test/mlir/matmul.mlir
```