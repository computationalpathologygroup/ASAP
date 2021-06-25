IF(PACKAGE_ON_INSTALL)
IF(WIN32)
  SET(APPS "\${CMAKE_INSTALL_PREFIX}/bin/ASAP.exe")
  SET(APPS_DEBUG "\${CMAKE_INSTALL_PREFIX}/bin/ASAP_d.exe")
  SET(LIBS "\${CMAKE_INSTALL_PREFIX}/bin/libEGL.dll" "\${CMAKE_INSTALL_PREFIX}/bin/Qt5OpenGL.dll" "\${CMAKE_INSTALL_PREFIX}/bin/plugins/filters/NucleiDetectionFilterPlugin.dll" "\${CMAKE_INSTALL_PREFIX}/bin/formats/openslidefileformat.dll")
  SET(LIBS_DEBUG "\${CMAKE_INSTALL_PREFIX}/bin/libEGLd.dll" "\${CMAKE_INSTALL_PREFIX}/bin/Qt5OpenGLd.dll" "\${CMAKE_INSTALL_PREFIX}/bin/plugins/filters/NucleiDetectionFilterPlugin_d.dll" "\${CMAKE_INSTALL_PREFIX}/bin/formats/openslidefileformat_d.dll")
  if(BUILD_IMAGEPROCESSING)
    get_target_property(opencv_dll_location opencv_world IMPORTED_LOCATION_RELEASE)
	if(NOT opencv_dll_location)
		get_target_property(opencv_dll_location opencv_core IMPORTED_LOCATION_RELEASE)
	endif(NOT opencv_dll_location)
    
    get_filename_component(opencv_dll_directory ${opencv_dll_location} DIRECTORY)
  endif(BUILD_IMAGEPROCESSING)
  MESSAGE(${OPENJPEG_INCLUDE_DIRS}/../../bin)
  SET(DIRS ${Qt5_DIR}/../../../bin ${opencv_dll_directory} ${OPENJPEG_INCLUDE_DIRS}/../../bin ${OPENSLIDE_INCLUDE_DIR}/../../bin ${PugiXML_INCLUDE_DIR}/../bin ${UNITTEST_INCLUDE_DIR}/../bin ${Boost_LIBRARY_DIRS} ${Boost_LIBRARY_DIR_RELEASE}/../bin ${Boost_LIBRARY_DIR_DEBUG}/../bin ${cpprestsdk_DIR}/../../bin)
  INCLUDE(InstallRequiredSystemLibraries)
  INSTALL(FILES ${Qt5_DIR}/../../../plugins/platforms/qwindows.dll DESTINATION bin/platforms CONFIGURATIONS Release)
  INSTALL(FILES ${Qt5_DIR}/../../../plugins/platforms/qwindowsd.dll DESTINATION bin/platforms CONFIGURATIONS Debug)
  IF(EXISTS ${Qt5_DIR}/../../../bin/libEGL.dll)
      INSTALL(FILES ${Qt5_DIR}/../../../bin/libEGL.dll DESTINATION bin CONFIGURATIONS Release)
  ENDIF(EXISTS ${Qt5_DIR}/../../../bin/libEGL.dll)
  IF(EXISTS ${Qt5_DIR}/../../../bin/libEGLd.dll)
      INSTALL(FILES ${Qt5_DIR}/../../../bin/libEGLd.dll DESTINATION bin CONFIGURATIONS Debug)
  ENDIF(EXISTS ${Qt5_DIR}/../../../bin/libEGLd.dll)  
  FILE(GLOB D3D_LIB ${Qt5_DIR}/../../../bin/d3dcompiler_*.dll)
  INSTALL(FILES ${D3D_LIB} DESTINATION bin)
  INSTALL(FILES ${Qt5_DIR}/../../../bin/Qt5OpenGL.dll DESTINATION bin CONFIGURATIONS Release)
  INSTALL(FILES ${Qt5_DIR}/../../../bin/Qt5OpenGLd.dll DESTINATION bin CONFIGURATIONS Debug)
  INSTALL(CODE "
   include(BundleUtilities)
   if(\"\${CMAKE_INSTALL_CONFIG_NAME}\" STREQUAL \"Debug\")
   fixup_bundle(\"${APPS_DEBUG}\"   \"${LIBS_DEBUG}\"   \"${DIRS}\")
   else()
   fixup_bundle(\"${APPS}\"   \"${LIBS}\"   \"${DIRS}\")
   endif()
   ")  
ELSEIF(APPLE)

   SET(APPS "\${CMAKE_INSTALL_PREFIX}/ASAP.app")  
   SET(prefix "ASAP.app/Contents")
   set(INSTALL_CMAKE_DIR "${prefix}/Resources")
   set(INSTALL_LIB_DIR "${prefix}/Frameworks")

   INSTALL(FILES "${Qt5_DIR}/../../../plugins/platforms/libqcocoa.dylib" DESTINATION "${prefix}/PlugIns/platforms")  
   
   set(LIBS "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/PlugIns/platforms/libqcocoa.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/formats/libopenslidefileformat.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/plugins/tools/libZoomToolPlugin.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/plugins/tools/libPanToolPlugin.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/plugins/filters/libNucleiDetectionFilterPlugin.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/plugins/workstationextension/libAnnotationPlugin.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/plugins/workstationextension/libFilterWorkstationExtensionPlugin.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/plugins/workstationextension/libVisualizationWorkstationExtensionPlugin.dylib")

   if(BUILD_IMAGEPROCESSING)
     get_target_property(opencv_dylib_location opencv_core IMPORTED_LOCATION_RELEASE)
   endif(BUILD_IMAGEPROCESSING)

   SET(DIRS ${CMAKE_BINARY_DIR}/ASAP
            ${CMAKE_BINARY_DIR}/multiresolutionimageinterface
            ${cpprestsdk_DIR}/../../lib 
            ${UNITTEST_INCLUDE_DIR}/../lib 
            ${Boost_LIBRARY_DIRS} 
            ${OPENSLIDE_INCLUDE_DIR}/../../lib 
            ${OPENSLIDE_INCLUDE_DIR}/../../bin 
            ${Qt5_DIR}/../.. 
            ${opencv_dylib_location} 
            ${OPENJPEG_INCLUDE_DIRS}/../../lib 
            ${OPENSLIDE_INCLUDE_DIR}/../../lib 
            ${PugiXML_INCLUDE_DIR}/../lib 
            ${UNITTEST_INCLUDE_DIR}/../lib 
            ${Boost_LIBRARY_DIRS} ${Boost_LIBRARY_DIR_RELEASE}/../lib 
            ${Boost_LIBRARY_DIR_DEBUG}/../lib)
            

   list(APPEND DIRS "${CMAKE_BINARY_DIR}")

   # Path used for searching by FIND_XXX(), with appropriate suffixes added
   if(CMAKE_BINARY_DIR)
     foreach(dir ${CMAKE_BINARY_DIR})
         list(APPEND DIRS "${dir}/bin" "${dir}/lib")
     endforeach()
   endif()

   file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
       "[Paths]\nPlugins = ${_qt_plugin_dir}\n")
   install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
       DESTINATION "${INSTALL_CMAKE_DIR}")

   include(InstallRequiredSystemLibraries)

   message(STATUS "DIRS: ${DIRS}")
   message(STATUS "LIBS: ${LIBS}")
   
   INSTALL(CODE "include(BundleUtilities)
     fixup_bundle(\"${APPS}\" \"${LIBS}\" \"${DIRS}\")")

ENDIF(WIN32)
ENDIF(PACKAGE_ON_INSTALL)
