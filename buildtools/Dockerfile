FROM ubuntu:18.04
ARG python_version=3.7
ARG only_multiresolutionimageinterface=false

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
        libunittest++-dev

RUN apt-get install -y -qq --no-install-recommends \
        python${python_version} \
        python${python_version}-dev \
        python3-pip \
        python3-setuptools

RUN rm /usr/bin/python3 
RUN ln -s /usr/bin/python${python_version} /usr/bin/python3    
        
RUN python3 -m pip install numpy==1.16.1
WORKDIR /root
RUN wget https://github.com/zeux/pugixml/archive/v1.9.tar.gz && tar xzf v1.9.tar.gz && rm v1.9.tar.gz
RUN wget ftp://dicom.offis.de/pub/dicom/offis/software/dcmtk/dcmtk364/dcmtk-3.6.4.tar.gz && tar xzf dcmtk-3.6.4.tar.gz && mv dcmtk-3.6.4 dcmtk && rm dcmtk-3.6.4.tar.gz
RUN git clone https://github.com/computationalpathologygroup/ASAP src
RUN mkdir build
WORKDIR /root/build      
COPY build_ASAP.sh /root/build/build_ASAP.sh
ENV BUILD_GUI=${only_multiresolutionimageinterface}
ENV BUILD_PYTHON_VERSION=${python_version}
RUN chmod a+rxwX ./build_ASAP.sh
CMD ./build_ASAP.sh
