#!/bin/bash
if [ "$1" != "" ]; then
    python_ver="$1"
else
    python_ver="3.8"
fi

if [ "$2" != "" ]; then
    python_ver_no_dot="$2"
else
    python_ver_no_dot="38"
fi

if [ "$3" != "" ]; then
    build_gui="$3"
else
    build_gui="false"
fi
ubuntu_version=$(grep 'DISTRIB_RELEASE' /etc/lsb-release | cut -d'=' -f2)
ubuntu_version_no_dots=$(echo ${ubuntu_version} | tr -d ".")
echo "Building ASAP with Python ${python_ver}; building GUI = ${build_gui}; on Ubuntu ${ubuntu_version}"
if [ "${build_gui}" = "true" ] ; then \
        cmake ../src -DPugiXML_INCLUDE_DIR=/root/pugixml-1.9/src/ -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                    -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                    -DBUILD_ASAP=TRUE -DBUILD_EXECUTABLES=TRUE -DBUILD_IMAGEPROCESSING=TRUE -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE -DCMAKE_BUILD_TYPE=Release \
                    -DDCMTKJPEG_INCLUDE_DIR=/root -DDCMTKJPEG_LIBRARY=/usr/lib/libijg8.so -DUNITTEST_INCLUDE_DIR=/usr/include/UnitTest++ \
                    -DPACKAGE_ON_INSTALL=TRUE -DBUILD_WORKLIST_INTERFACE=TRUE \
                    -DPYTHON_DEBUG_LIBRARY=/root/miniconda3/envs/build_python${python_ver}/lib/libpython${python_ver}.so \
                    -DPYTHON_LIBRARY_DEBUG=/root/miniconda3/envs/build_python${python_ver}/lib/libpython${python_ver}.so \
                    -DPYTHON_LIBRARY=/root/miniconda3/envs/build_python${python_ver}/lib/libpython${python_ver}.so \
                    -DPYTHON_INCLUDE_DIR=/root/miniconda3/envs/build_python${python_ver}/include/python${python_ver} \
                    -DPYTHON_EXECUTABLE=/root/miniconda3/envs/build_python${python_ver}/bin/python \
                    -DPYTHON_NUMPY_INCLUDE_DIR=/root/miniconda3/envs/build_python${python_ver}/lib/python${python_ver}/site-packages/numpy/core/include \
    ; else \
        echo "Skipping GUI..."
        cmake ../src -DPugiXML_INCLUDE_DIR=/root/pugixml-1.9/src/ -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                 -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                 -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE -DCMAKE_BUILD_TYPE=Release \
                 -DDCMTKJPEG_INCLUDE_DIR=/root -DDCMTKJPEG_LIBRARY=/usr/lib/libijg8.so -DPACKAGE_ON_INSTALL=TRUE \
                 -DPYTHON_DEBUG_LIBRARY=/root/miniconda3/envs/build_python${python_ver}/lib/libpython${python_ver}.so \
                 -DPYTHON_LIBRARY_DEBUG=/root/miniconda3/envs/build_python${python_ver}/lib/libpython${python_ver}.so \
                 -DPYTHON_LIBRARY=/root/miniconda3/envs/build_python${python_ver}/lib/libpython${python_ver}.so \
                 -DPYTHON_INCLUDE_DIR=/root/miniconda3/envs/build_python${python_ver}/include/python${python_ver} \
                 -DPYTHON_EXECUTABLE=/root/miniconda3/envs/build_python${python_ver}/bin/python \
                 -DPYTHON_NUMPY_INCLUDE_DIR=/root/miniconda3/envs/build_python${python_ver}/lib/python${python_ver}/site-packages/numpy/core/include \
    ; fi
export LD_LIBRARY_PATH=/root/miniconda3/envs/build_python${python_ver}/lib
make package

if [ "${build_gui}" = "true" ] ; then
        for file in *.deb; do
          outbasename="$(cut -d'-' -f1,2 <<<"$file")"
          mv $file /artifacts/${outbasename}-py${python_ver_no_dot}-Ubuntu${ubuntu_version_no_dots}.deb
        done;
else
        for file in *.deb; do
          outbasename="$(cut -d'-' -f1,2 <<<"$file")"
          mv $file /artifacts/${outbasename}-nogui-py${python_ver_no_dot}-Ubuntu${ubuntu_version_no_dots}.deb
        done;
fi