ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION}

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
        cmake \
        dpkg-dev \
        file \
        g++ \
        make \
        qt5-default \
        qtbase5-dev \
        qttools5-dev \
        libqt5opengl5-dev \
        git \
        libboost-all-dev \
        libopencv-dev \
        libdcmtk-dev \
        libopenjp2-7-dev \
        libopenslide-dev \
        wget \
        swig \
        libunittest++-dev \
        ca-certificates \
        nano

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN /root/miniconda3/bin/conda update --quiet -n base -c defaults conda && \
    /root/miniconda3/bin/conda create --quiet --name build_python3.6m python=3.6 && \
    /root/miniconda3/bin/conda install --quiet -n build_python3.6m numpy nomkl &&\
    /root/miniconda3/bin/conda create --quiet --name build_python3.7m python=3.7 && \
    /root/miniconda3/bin/conda install --quiet -n build_python3.7m numpy nomkl && \
    /root/miniconda3/bin/conda create --quiet --name build_python3.8 python=3.8 && \
    /root/miniconda3/bin/conda install --quiet -n build_python3.8 numpy nomkl

RUN ln -sd /root/miniconda3/envs/build_python3.7m/lib/python3.7 /root/miniconda3/envs/build_python3.7m/lib/python3.7m && \
    ln -sd /root/miniconda3/envs/build_python3.6m/lib/python3.6 /root/miniconda3/envs/build_python3.6m/lib/python3.6m

# Small hack to as libijg8 is installed in a different locations in Ubuntu 20.04
RUN ln -s /usr/lib/x86_64-linux-gnu/libijg8.so /usr/lib/libijg8.so; exit 0;

WORKDIR /root
RUN wget -q https://github.com/zeux/pugixml/archive/v1.9.tar.gz && tar xzf v1.9.tar.gz && rm v1.9.tar.gz && \
    wget -q ftp://dicom.offis.de/pub/dicom/offis/software/dcmtk/dcmtk364/dcmtk-3.6.4.tar.gz && tar xzf dcmtk-3.6.4.tar.gz && mv dcmtk-3.6.4 dcmtk && rm dcmtk-3.6.4.tar.gz
RUN git clone https://github.com/computationalpathologygroup/ASAP src &&\
    mkdir build

WORKDIR /root/build      
COPY build_ASAP.sh /root/build/build_ASAP.sh
RUN chmod a+rxwX ./build_ASAP.sh

CMD ./build_ASAP.sh