cmake_minimum_required(VERSION 2.8.9)
CMAKE_POLICY(SET CMP0020 NEW)
CMAKE_POLICY(SET CMP0043 NEW)
CMAKE_POLICY(SET CMP0053 NEW)

PROJECT(PathologyWorkstation)
INCLUDE_DIRECTORIES(
  ${PathologyWorkstation_SOURCE_DIR}/../
  ${Boost_INCLUDE_DIRS}
  ${OPENSLIDE_INCLUDE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${BASICFILTERS_BINARY_DIR}
)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
ADD_DEFINITIONS(${Boost_DEFINITIONS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(HEADERS
  pathologyworkstation.h
  PathologyViewer.h
  WSITileGraphicsItemCache.h
  MiniMap.h
  RenderThread.h
  RenderWorker.h
  TileManager.h
  PrefetchThread.h
  WSITileGraphicsItem.h
  UtilityFunctions.h
  ScaleBar.h
  interfaces/interfaces.h
  QtProgressMonitor.h
  ASAP_export.h
)

set(SOURCE 
  pathologyworkstation.cpp
  PathologyViewer.cpp
  WSITileGraphicsItemCache.cpp
  MiniMap.cpp
  RenderThread.cpp
  RenderWorker.cpp
  PrefetchThread.cpp
  TileManager.cpp
  WSITileGraphicsItem.cpp
  ScaleBar.cpp
  UtilityFunctions.cpp
  QtProgressMonitor.cpp
)

find_package(Qt5 COMPONENTS Core Widgets Gui OpenGL)
set(RESOURCE
resources/ASAP_resources.qrc
)
qt5_add_resources(RESOURCE_ADDED ${RESOURCE})      
ADD_LIBRARY(ASAPLib SHARED ${SOURCE} ${HEADERS} ${RESOURCE_ADDED})
QT5_USE_MODULES(ASAPLib Core Widgets OpenGL UiTools)
ADD_EXECUTABLE(ASAP WIN32 main.cpp PathologyWorkstation.rc ${RESOURCE_ADDED})

TARGET_LINK_LIBRARIES(ASAPLib multiresolutionimageinterface ${OpenCV_LIBS} basicfilters)
TARGET_LINK_LIBRARIES(ASAP ASAPLib)
SET_TARGET_PROPERTIES(ASAPLib PROPERTIES DEBUG_POSTFIX _d)
SET_TARGET_PROPERTIES(ASAP PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS ASAP
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(TARGETS ASAPLib
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES ${HEADERS} DESTINATION include/ASAP)

IF(WIN32)
  SET_TARGET_PROPERTIES(ASAP PROPERTIES FOLDER pathology/diag/ASAP)
  target_compile_definitions(ASAPLib PRIVATE -DASAP_EXPORTS)
  SET_TARGET_PROPERTIES(ASAPLib PROPERTIES FOLDER pathology/diag/ASAP)
ENDIF(WIN32)

# Add the basictools, annotation, filter plugins
ADD_SUBDIRECTORY(annotation)
ADD_SUBDIRECTORY(visualization)
ADD_SUBDIRECTORY(filters)
ADD_SUBDIRECTORY(basictools)