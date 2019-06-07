echo "Building ASAP with Python ${BUILD_PYTHON_VERSION}"
if [ "${BUILD_GUI}" = "false" ] ; then \
        cmake ../src -DPugiXML_INCLUDE_DIR=/root/pugixml-1.9/src/ -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                    -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                    -DBUILD_ASAP=TRUE -DBUILD_EXECUTABLES=TRUE -DBUILD_IMAGEPROCESSING=TRUE -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE -DBUILD_TESTS=TRUE -DCMAKE_BUILD_TYPE=Release \
                    -DDCMTKJPEG_INCLUDE_DIR=/root -DDCMTKJPEG_LIBRARY=/usr/lib/libijg8.so -DUNITTEST_INCLUDE_DIR=/usr/include/UnitTest++ \
                    -DUNITTEST_LIBRARY=/usr/lib/x86_64-linux-gnu/libUnitTest++.so -DUNITTEST_LIBRARY_DEBUG=/usr/lib/x86_64-linux-gnu/libUnitTest++.so -DPACKAGE_ON_INSTALL=TRUE \
    ; else \
    	echo "Skipping GUI..."
        cmake ../src -DPugiXML_INCLUDE_DIR=/root/pugixml-1.9/src/ -DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide \
                 -DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=/root/install \
                 -DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=TRUE -DCMAKE_BUILD_TYPE=Release \
                 -DDCMTKJPEG_INCLUDE_DIR=/root -DDCMTKJPEG_LIBRARY=/usr/lib/libijg8.so -DPACKAGE_ON_INSTALL=TRUE \
    ; fi
make package
if [ "${BUILD_GUI}" = "false" ] ; then \
        for file in *.deb; do mv $file /artifacts/$(basename $file .deb)-python${BUILD_PYTHON_VERSION}.deb; done \
; else \
        for file in *.deb; do mv $file /artifacts/$(basename $file .deb)-nogui-python${BUILD_PYTHON_VERSION}.deb; done \
; fi