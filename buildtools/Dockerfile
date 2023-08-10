ARG UBUNTU_VERSION=20.04

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
        qtbase5-dev \
        qttools5-dev \
        libqt5opengl5-dev \
        git \
        libopencv-dev \
        libdcmtk-dev \
        libopenjp2-7-dev \
        libopenslide-dev \
        libpugixml-dev \
        wget \
        gpg-agent \
        libunittest++-dev \
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
    /root/miniconda3/bin/conda create --quiet --name build_python3.9 python=3.9 && \
    /root/miniconda3/bin/conda install --quiet -n build_python3.9 numpy nomkl

RUN mkdir -p /root/swig/build && \
    mkdir -p /root/swig/install && \
    git clone https://github.com/GeertLitjens/swig /root/swig/src && \
    cd /root/swig/build && \
    cmake ../src -DWITH_PCRE=OFF -DCMAKE_INSTALL_PREFIX=/root/swig/install -DBISON_EXECUTABLE=/usr/bin/bison && \
    cmake --build . --config Release && \
    cmake --install .

# Small hack to as libijg8 is installed in a different locations in Ubuntu 20.04
RUN ln -s /usr/lib/x86_64-linux-gnu/libijg8.so /usr/lib/libijg8.so; exit 0;

WORKDIR /root
RUN wget -q https://dicom.offis.de/download/dcmtk/dcmtk367/dcmtk-3.6.7.tar.gz && tar xzf dcmtk-3.6.7.tar.gz && mv dcmtk-3.6.7 dcmtk && rm dcmtk-3.6.7.tar.gz

WORKDIR /root/build
COPY build_ASAP.sh /root/build/build_ASAP.sh
RUN chmod a+rxwX ./build_ASAP.sh

CMD ./build_ASAP.sh
