# HeteroCL Dialect Prototype

This is an out-of-tree [MLIR](https://mlir.llvm.org/) dialect for [HeteroCL](https://github.com/cornell-zhang/heterocl).

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
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_INSTALL_UTILS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=~/.venv/hcl-dev/bin/python3
   mkdir tools/mlir/python/dialects # Makefile bug
   make -j

   # Export the generated MLIR Python library
   export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core:${PYTHONPATH}
   # Add Pybind11 include path
   export CPLUS_INCLUDE_PATH=$(python3 -c "import pybind11;print(pybind11.get_include())"):${CPLUS_INCLUDE_PATH}
   # Maybe add Python3 include path, which is used by <Python.h>
   python3-config --includes
   export CPLUS_INCLUDE_PATH=/the/above/include/path:${CPLUS_INCLUDE_PATH}
   ```

### Build HeteroCL Dialect
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. Please firstly clone our repository.
```sh
git clone https://github.com/cornell-zhang/hcl-dialect-prototype.git
cd hcl-dialect-prototype
mkdir build && cd build
```

- Build without Python binding
```sh
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=OFF
make -j
```

- Build with Python binding: please set `-DPYTHON_BINDING=ON`. After building, export the generated HCL library to enable function call in Python.
```sh
export PYTHONPATH=build/tools/hcl/python_packages/hcl_core:${PYTHONPATH}
```

Lastly, you can use the following integration test to see whether your built dialect works properly.
```
cmake --build . --target check-hcl
```


## Run HeteroCL Dialect
```sh
# perform loop transformation passes
./bin/hcl-opt --opt ../test/compute/tiling.mlir

# generate C++ HLS code
./bin/hcl-translate --emit-hlscpp ../test/memory/buffer_conv.mlir
```

Or you can use our provided script to directly generate C++ HLS code from MLIR.

```sh
cd ../examples
make ../test/compute/tiling
```


## References
* [ScaleHLS](https://github.com/hanchenye/scalehls)
* [Torch-MLIR](https://github.com/llvm/torch-mlir)