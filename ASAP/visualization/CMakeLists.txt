PROJECT(VisualizationPlugin)
INCLUDE_DIRECTORIES(
  ${VisualizationPlugin_SOURCE_DIR}/../../
  ${Boost_INCLUDE_DIRS}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${ANNOTATION_BINARY_DIR}
)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_VISUALIZATIONPLUGIN)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET(VisualizationWorkstationExtensionPlugin_SOURCE
  VisualizationWorkstationExtensionPlugin.cpp
  VisualizationWorkstationExtensionPlugin.h
)

find_package(Qt5 COMPONENTS Core Widgets Gui)
set(VisualizationWorkstationExtensionPlugin_RESOURCE
resources/VisualizationWorkstationExtensionPlugin_resources.qrc
)
qt5_add_resources(VISUALIZATION_RESOURCE_ADDED ${VisualizationWorkstationExtensionPlugin_RESOURCE})
add_library(VisualizationWorkstationExtensionPlugin SHARED ${VisualizationWorkstationExtensionPlugin_SOURCE} ${VISUALIZATION_RESOURCE_ADDED})
qt5_use_modules(VisualizationWorkstationExtensionPlugin Core Widgets UiTools)
TARGET_LINK_LIBRARIES(VisualizationWorkstationExtensionPlugin ASAPLib multiresolutionimageinterface core annotation)
SET_TARGET_PROPERTIES(VisualizationWorkstationExtensionPlugin PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS VisualizationWorkstationExtensionPlugin
  RUNTIME DESTINATION bin/plugins/workstationextension
  LIBRARY DESTINATION bin/plugins/workstationextension
  ARCHIVE DESTINATION lib/plugins/workstationextension
)

INSTALL(FILES VisualizationWorkstationExtensionPlugin.h DESTINATION include/ASAP/plugins/workstationextension)

IF(WIN32)
  SET_TARGET_PROPERTIES(VisualizationWorkstationExtensionPlugin PROPERTIES FOLDER pathology/diag/ASAP/plugins/workstationextensions)   
ENDIF(WIN32)