# HeteroCL Dialect Prototype

This is an out-of-tree [MLIR](https://mlir.llvm.org/) dialect for [HeteroCL](https://github.com/cornell-zhang/heterocl).

## Building

### Preliminary tools
- gcc >= 5.4
- cmake >= 3.13.4

### Install LLVM 14.0.0
- Download LLVM from [llvm-project](https://github.com/llvm/llvm-project/releases/tag/llvmorg-14.0.0-rc1) or checkout the Github branch
```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout tags/llvmorg-14.0.0-rc1
```

- Build
   - Without Python binding
   ```sh
   mkdir build && cd build
   cmake -G "Unix Makefiles" ../llvm \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_INSTALL_UTILS=ON
   make -j
   ```
   - With Python binding: Please follow the [official guide](https://mlir.llvm.org/docs/Bindings/Python/#generating-_dialect_namespace_ops_genpy-wrapper-modules) to set up the environment. In the following, we set up a virtual environment called `hcl-dev` using Python venv, but we prefer you to install Anaconda3 and create an environment there. If you want to use your own Python environment, please specify the path for `-DPython3_EXECUTABLE`.
   ```sh
   # Create a virtual environment. Make sure you have installed Python3.
   which python3
   python3 -m venv ~/.venv/hcl-dev
   source ~/.venv/hcl-dev/bin/activate

   # It is recommended to upgrade pip
   python3 -m pip install --upgrade pip

   # Install required packages. Suppose you are inside the llvm-project folder.
   python3 -m pip install -r mlir/python/requirements.txt

   # Run cmake
   mkdir build && cd build
   cmake -G "Unix Makefiles" ../llvm \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_INSTALL_UTILS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=`which python3`
   mkdir tools/mlir/python/dialects # Makefile bug
   make -j

   # Export the generated MLIR Python library
   export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core:${PYTHONPATH}

   # Export the LLVM build directory
   export LLVM_BUILD_DIR=$(pwd)

   # To enable better backtracing for debugging,
   # we suggest setting the following system path
   export LLVM_SYMBOLIZER_PATH=$(pwd)/bin/llvm-symbolizer
   ```

### Build HeteroCL Dialect
This setup assumes that you have built LLVM and MLIR in `$LLVM_BUILD_DIR`. Please firstly clone our repository.
```sh
git clone https://github.com/cornell-zhang/hcl-dialect-prototype.git
cd hcl-dialect-prototype
mkdir build && cd build
```

- Build without Python binding
```sh
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=OFF
make -j
```

- Build with Python binding
```sh
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=ON \
   -DPython3_EXECUTABLE=~/.venv/hcl-dev/bin/python3
make -j

# Export the LD_LIBRARY_PATH for OpenSCoP library
export LD_LIBRARY_PATH=$(pwd)/openscop/lib:$LD_LIBRARY_PATH

# Export the generated HCL-MLIR Python library
export PYTHONPATH=$(pwd)/tools/hcl/python_packages/hcl_core:${PYTHONPATH}
```

Lastly, you can use the following integration test to see whether your built dialect works properly.
```
cmake --build . --target check-hcl
```


## Run HeteroCL Dialect
```sh
# perform loop transformation passes
./bin/hcl-opt -opt ../test/Transforms/compute/tiling.mlir

# generate C++ HLS code
./bin/hcl-opt -opt ../test/Transforms/compute/tiling.mlir | \
./bin/hcl-translate -emit-hlscpp

# generate OpenSCoP
./bin/hcl-opt -opt ../test/compute/tiling.mlir | \
./bin/hcl-translate --extract-scop-stmt

# run code on CPU
./bin/hcl-opt -jit ../test/Translation/mm.mlir
```

Or you can use our provided script to directly generate C++ HLS code from MLIR.

```sh
cd ../examples
make ../test/Transforms/compute/tiling
```

## Integrate with upstream HeteroCL frontend
Make sure you have correctly built the above HCL-MLIR dialect, and follow the instruction below.

```sh
# clone the HeteroCL repo
git clone https://github.com/chhzh123/heterocl.git heterocl-mlir
cd heterocl-mlir
git checkout hcl-mlir

# set up LLVM and cmake paths in Makefile.config
# You can reuse the built LLVM 14 above, just set
# LLVM_CONFIG = $BUILD_DIR/bin/llvm-config
# ...

# build frontend (for TVM IR)
# notice: no need to build if you use HCL-MLIR as IR
make -j

# export library
export HCL_HOME=$(pwd)
export PYTHONPATH=$HCL_HOME/python:$HCL_HOME/hlib/python:${PYTHONPATH}

# run tests
python3 tests/mlir/test_gemm.py
```

We retain the original Halide/TVM code and fully decoupled our frontend integration from the original HeteroCL implementation. All the MLIR frontend facilities are in the [`python/heterocl/mlir`](https://github.com/chhzh123/heterocl/tree/hcl-mlir/python/heterocl/mlir) folder. As a result, you can simply set the environment variable `HCLIR` to use different compilation flows. Examples are shown below.

```sh
# run original TVM flow (v1)
HCLIR=tvm python3 tests/mlir/test_gemm.py

# run HCL-MLIR flow (v2)
HCLIR=mlir python3 tests/mlir/test_gemm.py
```

Notice the integration is still in a very early stage, so not all the functionalities of the original HeteroCL are supported. If you experience any questions, please feel free to raise an issue.


## Coding Style

We follow [Google Style Guides](https://google.github.io/styleguide/) and use
* [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for C/C++
* [black](https://github.com/psf/black) and [pylint](https://pylint.org/) for Python


## References
* [ScaleHLS](https://github.com/hanchenye/scalehls)
* [Torch-MLIR](https://github.com/llvm/torch-mlir)
* [Polymer](https://github.com/kumasento/polymer)
