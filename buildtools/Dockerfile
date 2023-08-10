ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN export DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
        cmake \
        dpkg-dev \
        file \
        g++ \
        make \
        bison \
        git \
        libopencv-dev \
        libturbojpeg0-dev \
        libdcmtk-dev \
        libopenslide-dev \
        libpugixml-dev \
        libgl1-mesa-dev \
        wget \
        gpg-agent \
        ca-certificates \
        nano \
        software-properties-common \
        lsb-release \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y -qq --no-install-recommends \
        g++-10 \
        gcc-10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 50 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 50

# Ensure latest version of CMake
RUN apt purge -y --auto-remove cmake 
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && \
    apt install -y cmake

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN /root/miniconda3/bin/conda update --quiet -n base -c defaults conda && \
    /root/miniconda3/bin/conda create --quiet --name build python=3.9 && \
    /root/miniconda3/bin/conda install --quiet -n build numpy nomkl

RUN mkdir -p /root/swig/build && \
    mkdir -p /root/swig/install && \
    git clone https://github.com/GeertLitjens/swig /root/swig/src && \
    cd /root/swig/build && \
    cmake ../src -DWITH_PCRE=OFF -DCMAKE_INSTALL_PREFIX=/root/swig/install -DBISON_EXECUTABLE=/usr/bin/bison && \
    cmake --build . --config Release && \
    cmake --install .

RUN /root/miniconda3/bin/activate build && \
    /root/miniconda3/envs/build/bin/pip install aqtinstall && \
    /root/miniconda3/envs/build/bin/aqt install-qt --outputdir /root/qt linux desktop 6.5.2 gcc_64

RUN mkdir /root/openjpeg && \
    mkdir /root/openjpeg/build && \
    mkdir /root/openjpeg/install && \
    git clone https://github.com/uclouvain/openjpeg /root/openjpeg/src && \
    cd /root/openjpeg/build && \
    cmake ../src -DCMAKE_INSTALL_PREFIX=/root/openjpeg/install -DCMAKE_C_FLAGS=-fPIC -DBUILD_CODEC=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && \
    make install

WORKDIR /root/build
COPY build_ASAP.sh /root/build/build_ASAP.sh
RUN chmod a+rxwX ./build_ASAP.sh

# Workaround for weird CMake bug
RUN mv /root/miniconda3/envs/build/lib/libstdc++.so.6 /root/miniconda3/envs/build/lib/.libstdc++.so.6

CMD ./build_ASAP.sh
