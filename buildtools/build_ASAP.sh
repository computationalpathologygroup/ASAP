#!/bin/bash

if [ "$1" != "" ]; then
    build_gui="$1"
else
    build_gui="true"
fi
ubuntu_version=$(grep 'DISTRIB_RELEASE' /etc/lsb-release | cut -d'=' -f2)
ubuntu_version_no_dots=$(echo ${ubuntu_version} | tr -d ".")

echo "Building ASAP with Python ${python_ver}; building GUI = ${build_gui}; on Ubuntu ${ubuntu_version}"
git clone https://github.com/computationalpathologygroup/ASAP src
cd src
git checkout upgrading_dependencies
cd ..
mkdir build
cd build

if [ "${build_gui}" = "true" ] ; then \
        cmake ../src -DBUILD_ASAP=TRUE \
                     -DBUILD_EXECUTABLES=TRUE \
                     -DBUILD_IMAGEPROCESSING=TRUE \
                     -DBUILD_WORKLIST_INTERFACE=TRUE \
                     -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE \
                     -DCMAKE_BUILD_TYPE=Release \
                     -DPACKAGE_ON_INSTALL=TRUE \
                     -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                     -DCMAKE_INSTALL_PREFIX=/root/install \
                     -DSWIG_EXECUTABLE=/root/swig/install/bin/swig \
    ; else \
        echo "Skipping GUI..."
        cmake ../src -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                     -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE \
                     -DCMAKE_INSTALL_PREFIX=/root/install \
                     -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE \
                     -DCMAKE_BUILD_TYPE=Release \
                     -DPACKAGE_ON_INSTALL=TRUE \
                     -DSWIG_EXECUTABLE=/root/swig/install/bin/swig \
    ; fi
export LD_LIBRARY_PATH=/root/miniconda3/envs/build_python3.9/lib
make package

if [ "${build_gui}" = "true" ] ; then
        for file in *.deb; do
          outbasename="$(cut -d'-' -f1,2 <<<"$file")"
          mv $file /artifacts/${outbasename}-Ubuntu${ubuntu_version_no_dots}.deb
        done;
else
        for file in *.deb; do
          outbasename="$(cut -d'-' -f1,2 <<<"$file")"
          mv $file /artifacts/${outbasename}-nogui-Ubuntu${ubuntu_version_no_dots}.deb
        done;
fi
