cmake_minimum_required(VERSION 2.8.9)
CMAKE_POLICY(SET CMP0020 NEW)
CMAKE_POLICY(SET CMP0043 NEW)
PROJECT(PathologyWorkstation)
INCLUDE_DIRECTORIES(
  ${PathologyWorkstation_SOURCE_DIR}/../
  ${Boost_INCLUDE_DIRS}
  ${OPENSLIDE_INCLUDE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
ADD_DEFINITIONS(${Boost_DEFINITIONS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(HEADERS
  main.cpp
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
)

set(SOURCE 
  main.cpp
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
  PathologyWorkstation.rc
  QtProgressMonitor.cpp
)

IF(BUILD_USING_QT4)
  QT4_WRAP_CPP(PATHOLOHWORKSTATION_MOC pathologyworkstation.h PathologyViewer.h RenderThread.h PrefetchThread.h)
  set(RESOURCE
    resources/icons.qrc
  )
  qt4_add_resources(RESOURCE_ADDED ${RESOURCE})
  ADD_EXECUTABLE(ASAP WIN32 ${SOURCE} ${HEADERS} ${PATHOLOHWORKSTATION_MOC})
  TARGET_LINK_LIBRARIES(ASAP ${QT_LIBRARIES})
ELSE()
  find_package(Qt5 COMPONENTS Core Widgets Gui OpenGL)
  set(RESOURCE
    resources/ASAP_resources.qrc
  )
  qt5_add_resources(RESOURCE_ADDED ${RESOURCE})      
  ADD_EXECUTABLE(ASAP WIN32 ${SOURCE} ${HEADERS} ${RESOURCE_ADDED})
  QT5_USE_MODULES(ASAP Core Widgets OpenGL UiTools)
ENDIF()

TARGET_LINK_LIBRARIES(ASAP multiresolutionimageinterface ${OpenCV_LIBS} basicfilters)
SET_TARGET_PROPERTIES(ASAP PROPERTIES DEBUG_POSTFIX _d)
SET_TARGET_PROPERTIES(ASAP PROPERTIES ENABLE_EXPORTS TRUE)

INSTALL(TARGETS ASAP
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES ${HEADERS} DESTINATION include/ASAP)

IF(WIN32)
  target_compile_definitions(ASAP PRIVATE -DBUILD_PATHOLOGYWORKSTATION)
  SET_TARGET_PROPERTIES(ASAP PROPERTIES FOLDER pathology/diag/workstation)
ENDIF(WIN32)

# Add the basictools, annotation, filter plugins
ADD_SUBDIRECTORY(annotation)
ADD_SUBDIRECTORY(visualization)
ADD_SUBDIRECTORY(filters)
ADD_SUBDIRECTORY(basictools)