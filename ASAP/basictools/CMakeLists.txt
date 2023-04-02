set(CMAKE_AUTOMOC ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../lib")

set(ZoomTool_SOURCE 
  ZoomTool.h
  ZoomTool.cpp
)

set(PanTool_SOURCE 
  PanTool.h
  PanTool.cpp
)

qt6_add_resources(TOOL_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/resources/basictools_resources.qrc)
add_library(ZoomToolPlugin SHARED ${ZoomTool_SOURCE} ${TOOL_RESOURCES})
add_library(PanToolPlugin SHARED ${PanTool_SOURCE} ${TOOL_RESOURCES})

target_link_libraries(PanToolPlugin PRIVATE ASAPLib Qt6::Core Qt6::Widgets)
target_link_libraries(ZoomToolPlugin PRIVATE ASAPLib Qt6::Core Qt6::Widgets)
set_target_properties(PanToolPlugin PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(ZoomToolPlugin PROPERTIES DEBUG_POSTFIX _d)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")

  install(TARGETS PanToolPlugin ZoomToolPlugin
          BUNDLE DESTINATION . 
          RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/plugins/tools
          LIBRARY DESTINATION ${INSTALL_RUNTIME_DIR}/plugins/tools
          ARCHIVE DESTINATION ${INSTALL_CMAKE_DIR}/plugins/tools
  )

ELSE(APPLE)
  install(TARGETS PanToolPlugin ZoomToolPlugin
          RUNTIME DESTINATION bin/plugins/tools
          LIBRARY DESTINATION bin/plugins/tools
          ARCHIVE DESTINATION lib/plugins/tools
  )
ENDIF(APPLE)

install(FILES PanTool.h ZoomTool.h DESTINATION include/ASAP/plugins/tools)

IF(WIN32)
  set_target_properties(PanToolPlugin PROPERTIES FOLDER ASAP/plugins/tools)   
  set_target_properties(ZoomToolPlugin PROPERTIES FOLDER ASAP/plugins/tools)   
ENDIF(WIN32)
