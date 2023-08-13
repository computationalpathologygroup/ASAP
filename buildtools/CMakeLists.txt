IF(PACKAGE_ON_INSTALL)
IF(WIN32)
  SET(APPS "\${CMAKE_INSTALL_PREFIX}/bin/ASAP.exe")
  SET(APPS_DEBUG "\${CMAKE_INSTALL_PREFIX}/bin/ASAP_d.exe")
  SET(LIBS "\${CMAKE_INSTALL_PREFIX}/bin/plugins/filters/NucleiDetectionFilterPlugin.dll" "\${CMAKE_INSTALL_PREFIX}/bin/formats/openslidefileformat.dll" "\${CMAKE_INSTALL_PREFIX}/bin/formats/dicomfileformat.dll")
  SET(LIBS_DEBUG "\${CMAKE_INSTALL_PREFIX}/bin/plugins/filters/NucleiDetectionFilterPlugin_d.dll" "\${CMAKE_INSTALL_PREFIX}/bin/formats/openslidefileformat_d.dll" "\${CMAKE_INSTALL_PREFIX}/bin/formats/dicomfileformat_d.dll")
  if(BUILD_IMAGEPROCESSING)
      get_target_property(opencv_dll_location opencv_world IMPORTED_LOCATION_RELEASE)
	    if(NOT opencv_dll_location)
		      get_target_property(opencv_dll_location opencv_core IMPORTED_LOCATION_RELEASE)
	    endif(NOT opencv_dll_location)
    get_filename_component(opencv_dll_directory ${opencv_dll_location} DIRECTORY)
  endif(BUILD_IMAGEPROCESSING)
  
  SET(DIRS ${Qt6_DIR}/../../../bin ${opencv_dll_directory} ${OPENSLIDE_INCLUDE_DIR}/../../bin)
  
  if(BUILD_ASAP)
    INSTALL(FILES ${Qt6_DIR}/../../../plugins/platforms/qwindows.dll DESTINATION bin/platforms CONFIGURATIONS Release)
    INSTALL(FILES ${Qt6_DIR}/../../../plugins/platforms/qwindowsd.dll DESTINATION bin/platforms CONFIGURATIONS Debug)
  INSTALL(CODE "
   include(BundleUtilities)
   if(\"\${CMAKE_INSTALL_CONFIG_NAME}\" STREQUAL \"Debug\")
   fixup_bundle(\"${APPS_DEBUG}\"   \"${LIBS_DEBUG}\"   \"${DIRS}\")
   else()
   fixup_bundle(\"${APPS}\"   \"${LIBS}\"   \"${DIRS}\")
   endif()
   ")  
   ENDIF(BUILD_ASAP)
ELSEIF(APPLE)

   SET(APPS "\${CMAKE_INSTALL_PREFIX}/ASAP.app")  
   SET(prefix "ASAP.app/Contents")
   set(INSTALL_CMAKE_DIR "${prefix}/Resources")
   set(INSTALL_LIB_DIR "${prefix}/Frameworks")

   INSTALL(FILES "${Qt6_DIR}/../../../plugins/platforms/libqcocoa.dylib" DESTINATION "${prefix}/PlugIns/platforms")  
   INSTALL(FILES "${Qt6_DIR}/../../../plugins/platforms/libqminimal.dylib" DESTINATION "${prefix}/PlugIns/platforms")  
   INSTALL(FILES "${Qt6_DIR}/../../../plugins/platforms/libqoffscreen.dylib" DESTINATION "${prefix}/PlugIns/platforms")  
   INSTALL(FILES "${Qt6_DIR}/../../../plugins/platforms/libqwebgl.dylib" DESTINATION "${prefix}/PlugIns/platforms")  
   INSTALL(FILES "${DIAGPathology_SOURCE_DIR}/ASAP/icon.icns" DESTINATION "${prefix}/Resources")  
   INSTALL(FILES "${DIAGPathology_SOURCE_DIR}/ASAP/Info.plist" DESTINATION "${prefix}")  

   set(LIBS "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/PlugIns/platforms/libqcocoa.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/formats/libopenslidefileformat.dylib"
            "\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${prefix}/MacOS/formats/libdicomfileformat.dylib"
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
            ${DCMTK_DIR}/../../lib 
            ${UNITTEST_INCLUDE_DIR}/../lib 
            ${OPENSLIDE_INCLUDE_DIR}/../../lib 
            ${OPENSLIDE_INCLUDE_DIR}/../../bin 
            ${Qt6_DIR}/../.. 
            ${opencv_dylib_location} 
            ${OPENJPEG_INCLUDE_DIRS}/../../lib 
            ${OPENSLIDE_INCLUDE_DIR}/../../lib 
            ${PugiXML_INCLUDE_DIR}/../lib 
            ${UNITTEST_INCLUDE_DIR}/../lib )


   list(APPEND DIRS "${CMAKE_BINARY_DIR}")
   
   # Path used for searching by FIND_XXX(), with appropriate suffixes added
   if(CMAKE_BINARY_DIR)
     foreach(dir ${CMAKE_BINARY_DIR})
         list(APPEND DIRS "${dir}/bin" "${dir}/lib")
     endforeach()
   endif()

  find_package(Qt6 COMPONENTS Core Widgets Gui OpenGL UiTools Concurrent)

  # Install the plugins for each qt module
  set(REQ_QT_PLUGINS
          ${Qt6Core_PLUGINS}
          ${Qt6Widgets_PLUGINS}
          ${Qt6OpenGL_PLUGINS}
          ${Qt6UiTools_PLUGINS}
          ${Qt6Concurrent_PLUGINS}
          ${Qt6Gui_PLUGINS})
  foreach(plugin ${REQ_QT_PLUGINS})
      get_target_property(_qt_plugin_path ${plugin} LOCATION)
      get_filename_component(_qt_plugin_file "${_qt_plugin_path}" NAME)
      get_filename_component(_qt_plugin_type "${_qt_plugin_path}" PATH)
      get_filename_component(_qt_plugin_type "${_qt_plugin_type}" NAME)
      set(_qt_plugin_dest "${prefix}/PlugIns/${_qt_plugin_type}")
      install(FILES "${_qt_plugin_path}"
          DESTINATION "${_qt_plugin_dest}")
      set(QT_PLUGINS
              "${QT_PLUGINS};\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${_qt_plugin_dest}/${_qt_plugin_file}")
  endforeach()  

  list(APPEND LIBS ${QT_PLUGINS})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
      "[Paths]\nPlugins = ${_qt_plugin_dir}\n")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
      DESTINATION "${INSTALL_CMAKE_DIR}")

      message(STATUS "DIRS: ${DIRS}")
   message(STATUS "LIBS: ${LIBS}")
   
   INSTALL(CODE "include(BundleUtilities)
     fixup_bundle(\"${APPS}\" \"${LIBS}\" \"${DIRS}\")")
ELSE(WIN32)
    IF(BUILD_ASAP)
        INSTALL(FILES ${Qt6_DIR}/../../../plugins/platforms/libqxcb.so DESTINATION bin/platforms CONFIGURATIONS Release)
        INSTALL(FILES ${Qt6_DIR}/../../../lib/libQt6XcbQpa.so.6 DESTINATION lib CONFIGURATIONS Release)
        INSTALL(FILES ${Qt6_DIR}/../../../lib/libQt6XcbQpa.so.6.5.2 DESTINATION lib CONFIGURATIONS Release)
        INSTALL(FILES ${Qt6_DIR}/../../../plugins/platforms/libqxcbd.so DESTINATION bin/platforms CONFIGURATIONS Debug)
        INSTALL(FILES ${Qt6_DIR}/../../../lib/libQt6XcbQpad.so.6 DESTINATION lib CONFIGURATIONS Debug)
        INSTALL(FILES ${Qt6_DIR}/../../../lib/libQt6XcbQpa.so.6.5.2 DESTINATION lib CONFIGURATIONS Release)
        qt_generate_deploy_app_script(
            TARGET ASAP_browser
            OUTPUT_SCRIPT deploy_script
        )
        install(SCRIPT ${deploy_script})
    ENDIF(BUILD_ASAP)
ENDIF(WIN32)
ENDIF(PACKAGE_ON_INSTALL)