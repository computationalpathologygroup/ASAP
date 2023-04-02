set(HEADERS
    ASAP_Window.h
    PathologyViewer.h
    WSITileGraphicsItemCache.h
    MiniMap.h
    IOThread.h
    IOWorker.h
    TileManager.h
    PrefetchThread.h
    WSITileGraphicsItem.h
    UtilityFunctions.h
    ScaleBar.h
    interfaces/interfaces.h
    QtProgressMonitor.h
)

set(SOURCE 
    ASAP_Window.cpp
    PathologyViewer.cpp
    WSITileGraphicsItemCache.cpp
    MiniMap.cpp
    IOThread.cpp
    IOWorker.cpp
    PrefetchThread.cpp
    TileManager.cpp
    WSITileGraphicsItem.cpp
    ScaleBar.cpp
    QtProgressMonitor.cpp
)

set(RESOURCE
    resources/ASAP_resources.qrc
)

qt6_add_resources(RESOURCE_ADDED ${RESOURCE})      
add_library(ASAPLib SHARED ${SOURCE} ${HEADERS} ${RESOURCE_ADDED})
generate_export_header(ASAPLib)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  add_executable(ASAP MACOSX_BUNDLE main.cpp ASAP.rc ${RESOURCE_ADDED} ${DIAGPathology_SOURCE_DIR}/ASAP/icon.icns)

ELSE(APPLE)
  add_executable(ASAP WIN32 main.cpp ASAP.rc ${RESOURCE_ADDED})
ENDIF(APPLE)

target_link_libraries(ASAPLib PUBLIC basicfilters PRIVATE multiresolutionimageinterface Qt6::Widgets Qt6::Core Qt6::UiTools)
target_include_directories(ASAPLib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/ASAP> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/ASAP>)
target_link_libraries(ASAP PRIVATE ASAPLib Qt6::Core Qt6::Widgets)
if(WIN32)
target_compile_options(ASAP PRIVATE "/wd26812;")
target_compile_options(ASAPLib PRIVATE "/wd26812;")
endif(WIN32)
set_target_properties(ASAPLib PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(ASAP PROPERTIES DEBUG_POSTFIX _d)

IF(APPLE)
    install(TARGETS ASAP
            BUNDLE DESTINATION . 
            RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
            LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
            ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
    )

    INSTALL(TARGETS ASAPLib
            BUNDLE DESTINATION . 
            RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
            LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
            ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
    )
ELSE(APPLE)
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
ENDIF(APPLE)
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
ADD_SUBDIRECTORY(worklist_interface)
