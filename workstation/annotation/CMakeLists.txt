PROJECT(AnnotationPlugin)
INCLUDE_DIRECTORIES(
  ${AnnotationPlugin_SOURCE_DIR}/../../
  ${Boost_INCLUDE_DIRS}
  ${PugiXML_INCLUDE_DIR}
)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_PATHOLOGYANNOTATION)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET(ANNOTATION_HEADERS
  AnnotationWorkstationExtensionPlugin.h
  Annotation.h
  AnnotationGroup.h
  AnnotationList.h
  AnnotationService.h
  DotAnnotationTool.h
  PolyAnnotationTool.h
  SplineAnnotationTool.h
  QtAnnotation.h
  PolyQtAnnotation.h
  DotQtAnnotation.h
  XmlRepository.h
  Repository.h
)

SET(ANNOTATION_SOURCE
  AnnotationWorkstationExtensionPlugin.cpp
  Annotation.cpp
  AnnotationGroup.cpp
  AnnotationList.cpp
  AnnotationService.cpp
  DotAnnotationTool.cpp
  PolyAnnotationTool.cpp
  SplineAnnotationTool.cpp
  QtAnnotation.cpp
  PolyQtAnnotation.cpp
  DotQtAnnotation.cpp
  XmlRepository.cpp
  Repository.cpp
)

IF(BUILD_USING_QT4)
  set(RESOURCE
    resources/icons.qrc
  )
  qt4_add_resources(RESOURCE_ADDED ${RESOURCE})
  add_library(AnnotationPlugin SHARED ${ANNOTATION_SOURCE} ${ANNOTATION_HEADERS} ${ANNOTATION_RESOURCE_ADDED})
  TARGET_LINK_LIBRARIES(AnnotationPlugin ${QT_LIBRARIES})
ELSE()
  find_package(Qt5 COMPONENTS Core Widgets Gui)
  set(RESOURCE
    resources/icons.qrc
  )
  qt5_add_resources(ANNOTATION_RESOURCE_ADDED ${RESOURCE})
  add_library(AnnotationPlugin SHARED ${ANNOTATION_SOURCE} ${ANNOTATION_HEADERS} ${ANNOTATION_RESOURCE_ADDED})
  qt5_use_modules(AnnotationPlugin Core Widgets UiTools)
ENDIF()
TARGET_LINK_LIBRARIES(AnnotationPlugin pugixml ASAP core)
SET_TARGET_PROPERTIES(AnnotationPlugin PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS AnnotationPlugin
  RUNTIME DESTINATION bin/plugins/workstationextension
  LIBRARY DESTINATION lib/plugins/workstationextension
  ARCHIVE DESTINATION lib/plugins/workstationextension
)

INSTALL(FILES ${ANNOTATION_HEADERS} DESTINATION include/workstation/plugins/workstationextension)

IF(WIN32)
  SET_TARGET_PROPERTIES(AnnotationPlugin PROPERTIES FOLDER pathology/diag/workstation/plugins/workstationextensions)   
ENDIF(WIN32)