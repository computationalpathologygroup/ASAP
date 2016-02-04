PROJECT(AnnotationPlugin)
INCLUDE_DIRECTORIES(
  ${Annotation_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${PugiXML_INCLUDE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_PATHOLOGYANNOTATIONPLUGIN)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET(ANNOTATION_PLUGIN_HEADERS
  AnnotationWorkstationExtensionPlugin.h
  DotAnnotationTool.h
  PolyAnnotationTool.h
  PointSetAnnotationTool.h
  SplineAnnotationTool.h
  QtAnnotation.h
  QtAnnotationGroup.h
  PolyQtAnnotation.h
  PointSetQtAnnotation.h
  DotQtAnnotation.h
)

SET(ANNOTATION_PLUGIN_SOURCE
  AnnotationWorkstationExtensionPlugin.cpp
  DotAnnotationTool.cpp
  PolyAnnotationTool.cpp
  PointSetAnnotationTool.cpp
  SplineAnnotationTool.cpp
  QtAnnotation.cpp
  QtAnnotationGroup.cpp
  PolyQtAnnotation.cpp
  PointSetQtAnnotation.cpp
  DotQtAnnotation.cpp
)

IF(BUILD_USING_QT4)
  set(ANNOTATION_WORKSTATION_EXTENSION_RESOURCE
    resources/icons.qrc
  )
  qt4_add_resources(ANNOTATION_RESOURCE_ADDED ${ANNOTATION_WORKSTATION_EXTENSION_RESOURCE})
  add_library(AnnotationPlugin SHARED ${ANNOTATION_PLUGIN_SOURCE} ${ANNOTATION_PLUGIN_HEADERS} ${ANNOTATION_RESOURCE_ADDED})
  TARGET_LINK_LIBRARIES(AnnotationPlugin ${QT_LIBRARIES})
ELSE()
  find_package(Qt5 COMPONENTS Core Widgets Gui)
  set(ANNOTATION_WORKSTATION_EXTENSION_RESOURCE
    resources/AnnotationWorkstationExtensionPlugin_resources.qrc
  )
  qt5_add_resources(ANNOTATION_RESOURCE_ADDED ${ANNOTATION_WORKSTATION_EXTENSION_RESOURCE})
  add_library(AnnotationPlugin SHARED ${ANNOTATION_PLUGIN_SOURCE} ${ANNOTATION_PLUGIN_HEADERS} ${ANNOTATION_RESOURCE_ADDED})
  qt5_use_modules(AnnotationPlugin Core Widgets UiTools)
ENDIF()
TARGET_LINK_LIBRARIES(AnnotationPlugin pugixml ASAP core annotation)
SET_TARGET_PROPERTIES(AnnotationPlugin PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS AnnotationPlugin
  RUNTIME DESTINATION bin/plugins/workstationextension
  LIBRARY DESTINATION bin/plugins/workstationextension
  ARCHIVE DESTINATION lib/plugins/workstationextension
)

INSTALL(FILES ${ANNOTATION_HEADERS} DESTINATION include/workstation/plugins/workstationextension)

IF(WIN32)
  SET_TARGET_PROPERTIES(AnnotationPlugin PROPERTIES FOLDER pathology/diag/workstation/plugins/workstationextensions)   
ENDIF(WIN32)