# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

FROM centos:latest
ENV DEBIAN_FRONTEND=noninteractive

# install essentials
RUN yum update -y && yum install -y python3-devel git wget vim gdb gcc gcc-c++ kernel-devel make

# install cmake
wget https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3.tar.gz
tar -xzvf cmake-3.23.3.tar.gz
cd cmake-3.23.3
./bootstrap
make -j`nproc`
export PATH=$PATH:cmake-3.23.3/bin

# install conda env
WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    eval "$(~/miniconda/bin/conda shell.bash hook)" && \
    conda create --name hcl-dev python=3.7 -y && \
    conda activate hcl-dev

RUN cd /root/ && git clone https://github.com/llvm/llvm-project.git && \
    cd llvm-project && \
    git checkout tags/llvmorg-15.0.0-rc1 && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r mlir/python/requirements.txt && \
    mkdir build && cd build && \
    cmake -G "Unix Makefiles" ../llvm -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3` && \
    make -j`nproc`

# build HeteroCL dialect
# use your own token value: https://www.shanebart.com/clone-repo-using-token/ 
RUN cd /root/ && export TOKEN="Username:PersonalToken" && \
    export BUILD_DIR=/root/llvm-project/build && \
    export PREFIX=/root/llvm-project/build && \
    git clone --recursive https://$TOKEN@github.com/cornell-zhang/hcl-dialect.git && \
    cd hcl-dialect && mkdir build && cd build && \
    cmake -G "Unix Makefiles" .. \
        -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
        -DPYTHON_BINDING=ON \
        -DPython3_EXECUTABLE=`which python3` && \
    make -j`nproc`
ENV PYTHONPATH /root/hcl-dialect/build/tools/hcl/python_packages/hcl_core:${PYTHONPATH}

# test
RUN cd /root/hcl-dialect/build && \
   cmake --build . --target check-hcl
