set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(HEADERS
    ASAP_Window.h
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
	documents/Document.h
	documents/DocumentWindow.h
)

set(SOURCE 
    ASAP_Window.cpp
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
	documents/Document.cpp
	documents/DocumentWindow.cpp
)

find_package(Qt5 COMPONENTS Core Widgets Gui OpenGL UiTools)
set(RESOURCE
    resources/ASAP_resources.qrc
)

qt5_add_resources(RESOURCE_ADDED ${RESOURCE})      
add_library(ASAPLib SHARED ${SOURCE} ${HEADERS} ${RESOURCE_ADDED})
generate_export_header(ASAPLib)

add_executable(ASAP WIN32 main.cpp ASAP.rc ${RESOURCE_ADDED})

target_link_libraries(ASAPLib PUBLIC basicfilters PRIVATE multiresolutionimageinterface Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::UiTools)
target_include_directories(ASAPLib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/ASAP> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/ASAP>)
target_link_libraries(ASAP PRIVATE ASAPLib Qt5::Core Qt5::Widgets)
set_target_properties(ASAPLib PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(ASAP PROPERTIES DEBUG_POSTFIX _d)

install(TARGETS ASAP
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

install(FILES ${HEADERS} DESTINATION include/ASAP)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/asaplib_export.h DESTINATION include/ASAP)

IF(WIN32)
  set_target_properties(ASAP PROPERTIES FOLDER ASAP)
  target_compile_definitions(ASAPLib PRIVATE -DASAP_EXPORTS)
  set_target_properties(ASAPLib PROPERTIES FOLDER ASAP)
ENDIF(WIN32)

# Add the basictools, annotation, filter plugins
ADD_SUBDIRECTORY(annotation)
ADD_SUBDIRECTORY(visualization)
ADD_SUBDIRECTORY(filters)
ADD_SUBDIRECTORY(basictools)
