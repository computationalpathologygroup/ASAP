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
mkdir build
cd build

if [ "${build_gui}" = "true" ] ; then \
        cmake ../src -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                    -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                    -DBUILD_ASAP=TRUE -DBUILD_EXECUTABLES=TRUE -DBUILD_IMAGEPROCESSING=TRUE -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE -DCMAKE_BUILD_TYPE=Release \
                    -DDCMTKJPEG_INCLUDE_DIR=/root -DDCMTKJPEG_LIBRARY=/usr/lib/libijg8.so -DUNITTEST_INCLUDE_DIR=/usr/include/UnitTest++ \
                    -DPACKAGE_ON_INSTALL=TRUE -DBUILD_WORKLIST_INTERFACE=TRUE \
                    -DSWIG_EXECUTABLE=/root/swig/install/bin/swig \
                    -DPYTHON_DEBUG_LIBRARY=/root/miniconda3/envs/build_python3.9/lib/libpython3.so \
                    -DPYTHON_LIBRARY_DEBUG=/root/miniconda3/envs/build_python3.9/lib/libpython3.so \
                    -DPYTHON_LIBRARY=/root/miniconda3/envs/build_python3.9/lib/libpython3.so \
                    -DPYTHON_INCLUDE_DIR=/root/miniconda3/envs/build_python3.9/include/python3.9 \
                    -DPYTHON_EXECUTABLE=/root/miniconda3/envs/build_python3.9/bin/python \
                    -DPYTHON_NUMPY_INCLUDE_DIR=/root/miniconda3/envs/build_python3.9/lib/python3.9/site-packages/numpy/core/include \
    ; else \
        echo "Skipping GUI..."
        cmake ../src -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                 -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                 -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE -DCMAKE_BUILD_TYPE=Release \
                 -DDCMTKJPEG_INCLUDE_DIR=/root -DDCMTKJPEG_LIBRARY=/usr/lib/libijg8.so -DPACKAGE_ON_INSTALL=TRUE \
                    -DSWIG_EXECUTABLE=/root/swig/install/bin/swig \
                    -DPYTHON_DEBUG_LIBRARY=/root/miniconda3/envs/build_python3.9/lib/libpython3.so \
                    -DPYTHON_LIBRARY_DEBUG=/root/miniconda3/envs/build_python3.9/lib/libpython3.so \
                    -DPYTHON_LIBRARY=/root/miniconda3/envs/build_python3.9/lib/libpython3.so \
                    -DPYTHON_INCLUDE_DIR=/root/miniconda3/envs/build_python3.9/include/python3.9 \
                    -DPYTHON_EXECUTABLE=/root/miniconda3/envs/build_python3.9/bin/python \
                    -DPYTHON_NUMPY_INCLUDE_DIR=/root/miniconda3/envs/build_python3.9/lib/python3.9/site-packages/numpy/core/include \
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
