#!/bin/bash
if [ "$1" != "" ]; then
    build_gui="$1"
else
    build_gui="true"
fi
ubuntu_version=$(grep 'DISTRIB_RELEASE' /etc/lsb-release | cut -d'=' -f2)
ubuntu_version_no_dots=$(echo ${ubuntu_version} | tr -d ".")

echo "Building ASAP; building GUI = ${build_gui}; on Ubuntu ${ubuntu_version}"
git clone https://github.com/computationalpathologygroup/ASAP src
mkdir build
cd build

if [ "${build_gui}" = "true" ] ; then \
        cmake ../src -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide -DOpenJPEG_DIR=/root/openjpeg/install/lib/cmake/openjpeg-2.5 \
                     -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                     -DBUILD_ASAP=TRUE -DBUILD_EXECUTABLES=TRUE -DBUILD_IMAGEPROCESSING=TRUE -DCMAKE_BUILD_TYPE=Release \
                     -DPACKAGE_ON_INSTALL=TRUE -DBUILD_WORKLIST_INTERFACE=TRUE \
                     -DSWIG_EXECUTABLE=/root/swig/install/bin/swig \
                     -DQt6_DIR=/root/qt/6.5.2/gcc_64/lib/cmake/Qt6 \
                     -DQt6GuiTools_DIR=/root/qt/6.5.2/gcc_64/lib/cmake/Qt6GuiTools \
                     -DPython3_ROOT_DIR=/root/miniconda3/envs/build \
    ; else \
        echo "Skipping GUI..."
        cmake ../src -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide -DOpenJPEG_DIR=/root/openjpeg/install/lib/cmake/openjpeg-2.5 \
                     -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                     -DCMAKE_BUILD_TYPE=Release \
                     -DPACKAGE_ON_INSTALL=TRUE \
                     -DSWIG_EXECUTABLE=/root/swig/install/bin/swig \
                     -DPython3_ROOT_DIR=/root/miniconda3/envs/build \
    ; fi
export LD_LIBRARY_PATH=/root/miniconda3/envs/build/lib
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
