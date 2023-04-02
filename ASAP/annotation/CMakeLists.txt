set(CMAKE_AUTOMOC ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../lib")

SET(ANNOTATION_PLUGIN_HEADERS
    AnnotationWorkstationExtensionPlugin.h
    AnnotationTool.h
    DotAnnotationTool.h
    PolyAnnotationTool.h
    MeasurementAnnotationTool.h
    RectangleAnnotationTool.h
    PointSetAnnotationTool.h
    SplineAnnotationTool.h
    QtAnnotation.h
    QtAnnotationGroup.h
    MeasurementQtAnnotation.h
    PolyQtAnnotation.h
    PointSetQtAnnotation.h
    DotQtAnnotation.h
)

SET(ANNOTATION_PLUGIN_SOURCE
    AnnotationTool.cpp
    AnnotationWorkstationExtensionPlugin.cpp
    MeasurementAnnotationTool.cpp
    RectangleAnnotationTool.cpp
    DotAnnotationTool.cpp
    PolyAnnotationTool.cpp
    PointSetAnnotationTool.cpp
    SplineAnnotationTool.cpp
    QtAnnotation.cpp
    QtAnnotationGroup.cpp
    PolyQtAnnotation.cpp
    MeasurementQtAnnotation.cpp
    PointSetQtAnnotation.cpp
    DotQtAnnotation.cpp
)

set(ANNOTATION_WORKSTATION_EXTENSION_RESOURCE
    resources/AnnotationWorkstationExtensionPlugin_resources.qrc
)
qt6_add_resources(ANNOTATION_RESOURCE_ADDED ${ANNOTATION_WORKSTATION_EXTENSION_RESOURCE})
add_library(AnnotationPlugin SHARED ${ANNOTATION_PLUGIN_SOURCE} ${ANNOTATION_PLUGIN_HEADERS} ${ANNOTATION_RESOURCE_ADDED})

generate_export_header(AnnotationPlugin)
target_link_libraries(AnnotationPlugin PUBLIC ASAPLib core PRIVATE multiresolutionimageinterface core annotation Qt6::Core Qt6::Widgets Qt6:UiTools)
target_include_directories(AnnotationPlugin PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/ASAP/plugins/workstationextension>)
set_target_properties(AnnotationPlugin PROPERTIES DEBUG_POSTFIX _d)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")

  install(TARGETS AnnotationPlugin
          BUNDLE DESTINATION . 
          RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/plugins/workstationextension
          LIBRARY DESTINATION ${INSTALL_RUNTIME_DIR}/plugins/workstationextension
          ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}/plugins/workstationextension
  )
ELSE(APPLE)
  install(TARGETS AnnotationPlugin
            RUNTIME DESTINATION bin/plugins/workstationextension
            LIBRARY DESTINATION bin/plugins/workstationextension
            ARCHIVE DESTINATION lib/plugins/workstationextension
  )
ENDIF(APPLE)

install(FILES ${ANNOTATION_HEADERS} DESTINATION include/ASAP/plugins/workstationextension)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/annotationplugin_export.h DESTINATION include/ASAP/plugins/workstationextension)

IF(WIN32)
  SET_TARGET_PROPERTIES(AnnotationPlugin PROPERTIES FOLDER ASAP/plugins/workstationextensions)   
ENDIF(WIN32)
