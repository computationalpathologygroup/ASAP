set(CMAKE_AUTOMOC ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../lib")

set(VisualizationWorkstationExtensionPlugin_SOURCE
    VisualizationWorkstationExtensionPlugin.cpp
    VisualizationWorkstationExtensionPlugin.h
)

set(VisualizationWorkstationExtensionPlugin_RESOURCE
    resources/VisualizationWorkstationExtensionPlugin_resources.qrc
)

qt6_add_resources(VISUALIZATION_RESOURCE_ADDED ${VisualizationWorkstationExtensionPlugin_RESOURCE})
add_library(VisualizationWorkstationExtensionPlugin SHARED ${VisualizationWorkstationExtensionPlugin_SOURCE} ${VISUALIZATION_RESOURCE_ADDED})
target_link_libraries(VisualizationWorkstationExtensionPlugin PRIVATE ASAPLib multiresolutionimageinterface annotation Qt6::Core Qt6::Widgets Qt6::UiTools)
set_target_properties(VisualizationWorkstationExtensionPlugin PROPERTIES DEBUG_POSTFIX _d)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")

  install(TARGETS VisualizationWorkstationExtensionPlugin
          BUNDLE DESTINATION . 
          RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/plugins/workstationextension
          LIBRARY DESTINATION ${INSTALL_RUNTIME_DIR}/plugins/workstationextension
          ARCHIVE DESTINATION ${INSTALL_CMAKE_DIR}/plugins/workstationextension
  )

ELSE(APPLE)

  install(TARGETS VisualizationWorkstationExtensionPlugin
          RUNTIME DESTINATION bin/plugins/workstationextension
          LIBRARY DESTINATION bin/plugins/workstationextension
          ARCHIVE DESTINATION lib/plugins/workstationextension
  )
ENDIF(APPLE)
install(FILES VisualizationWorkstationExtensionPlugin.h DESTINATION include/ASAP/plugins/workstationextension)

IF(WIN32)
  set_target_properties(VisualizationWorkstationExtensionPlugin PROPERTIES FOLDER ASAP/plugins/workstationextensions)   
ENDIF(WIN32)
