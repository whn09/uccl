FROM lmsysorg/sglang:latest

# Base image includes: CUDA 12.9.1, PyTorch 2.9.1, GDRCopy 2.5.1, NCCL 2.27.3

# Install EFA dependencies + UCCL-EP build deps + Mooncake build deps
RUN apt-get update && apt-get install -y \
    libnuma-dev libgoogle-glog-dev libnl-3-dev libnl-route-3-dev \
    pci.ids environment-modules tcl \
    libgflags-dev libjsoncpp-dev libibverbs-dev libboost-all-dev \
    libcurl4-openssl-dev libyaml-cpp-dev libgtest-dev pybind11-dev \
    python3-dev libssl-dev libunwind-dev patchelf

# AWS EFA Installer v1.47.0
RUN cd /tmp \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz \
    && tar -xf aws-efa-installer-1.47.0.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod -g --no-verify \
    && rm -rf /tmp/aws-efa-installer-1.47.0.tar.gz /tmp/aws-efa-installer \
    && rm -rf /var/lib/apt/lists/*

# Upgrade SGLang to main branch (for Kimi-K2.5 support)
RUN cd /tmp && git clone --depth 1 https://github.com/sgl-project/sglang.git \
    && cd sglang/python && pip install -e ".[all]" \
    && rm -rf /tmp/sglang/.git

# Build yalantinglibs (Mooncake dependency)
RUN cd /tmp && git clone --depth 1 https://github.com/alibaba/yalantinglibs.git \
    && cd yalantinglibs && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) && make install \
    && rm -rf /tmp/yalantinglibs

# Build Mooncake with EFA support (PR #1523 branch)
RUN cd /tmp \
    && git clone --depth 1 -b efa-tcp-cuda-support https://github.com/whn09/Mooncake.git \
    && cd Mooncake && git submodule update --init --recursive \
    && mkdir build && cd build \
    && cmake .. \
        -DUSE_EFA=ON \
        -DUSE_CUDA=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_PREFIX_PATH="/opt/amazon/efa" \
        -DCMAKE_LIBRARY_PATH="/opt/amazon/efa/lib" \
        -DCMAKE_INCLUDE_PATH="/opt/amazon/efa/include" \
    && make -j$(nproc) \
    && cp mooncake-integration/engine.cpython-*.so ../mooncake-wheel/mooncake/ \
    && cp mooncake-asio/libasio.so ../mooncake-wheel/mooncake/ \
    && pip install -e ../mooncake-wheel --no-build-isolation \
    && rm -rf /tmp/Mooncake/.git

# Fix CuDNN compatibility with PyTorch 2.9.1 (need >= 9.15)
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# Fix FlashInfer version mismatch
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

# Set environment for EFA
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV NCCL_SOCKET_IFNAME=enp71s0

WORKDIR /workspace/uccl/ep/deep_ep_wrapper/
