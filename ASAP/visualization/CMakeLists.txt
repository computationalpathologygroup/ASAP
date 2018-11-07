set(CMAKE_AUTOMOC ON)

set(VisualizationWorkstationExtensionPlugin_SOURCE
    VisualizationWorkstationExtensionPlugin.cpp
    VisualizationWorkstationExtensionPlugin.h
)

set(VisualizationWorkstationExtensionPlugin_RESOURCE
    resources/VisualizationWorkstationExtensionPlugin_resources.qrc
)

qt5_add_resources(VISUALIZATION_RESOURCE_ADDED ${VisualizationWorkstationExtensionPlugin_RESOURCE})
add_library(VisualizationWorkstationExtensionPlugin SHARED ${VisualizationWorkstationExtensionPlugin_SOURCE} ${VISUALIZATION_RESOURCE_ADDED})
target_link_libraries(VisualizationWorkstationExtensionPlugin PRIVATE ASAPLib multiresolutionimageinterface annotation Qt5::Core Qt5::Widgets Qt5::UiTools)
set_target_properties(VisualizationWorkstationExtensionPlugin PROPERTIES DEBUG_POSTFIX _d)

install(TARGETS VisualizationWorkstationExtensionPlugin
        RUNTIME DESTINATION bin/plugins/workstationextension
        LIBRARY DESTINATION bin/plugins/workstationextension
        ARCHIVE DESTINATION lib/plugins/workstationextension
)

install(FILES VisualizationWorkstationExtensionPlugin.h DESTINATION include/ASAP/plugins/workstationextension)

IF(WIN32)
  set_target_properties(VisualizationWorkstationExtensionPlugin PROPERTIES FOLDER ASAP/plugins/workstationextensions)   
ENDIF(WIN32)