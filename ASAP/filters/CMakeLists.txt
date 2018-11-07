set(CMAKE_AUTOMOC ON)
set(FilterWorkstationExtensionPlugin_SOURCE
    FilterWorkstationExtensionPlugin.cpp
    FilterWorkstationExtensionPlugin.h
    FilterThread.h
    FilterThread.cpp
    FilterDockWidget.h
    FilterDockWidget.cpp
)

set(ColorDeconvolutionPlugin_SOURCE 
    ColorDeconvolutionFilterPlugin.h
    ColorDeconvolutionFilterPlugin.cpp
)

set(NucleiDetection_SOURCE 
    NucleiDetectionFilterPlugin.h
    NucleiDetectionFilterPlugin.cpp
)

qt5_add_resources(COLORDECONVOLUTION_RESOURCE ${CMAKE_CURRENT_SOURCE_DIR}/resources/ColorDeconvolutionFilterPlugin.qrc)
qt5_add_resources(FILTERWORKSTATIONEXTENSIONPLUGIN_RESOURCE ${CMAKE_CURRENT_SOURCE_DIR}/resources/FilterWorkstationExtensionPlugin_resources.qrc)

if(BUILD_IMAGEPROCESSING)
  qt5_add_resources(NUCLEIDETECTION_RESOURCE ${CMAKE_CURRENT_SOURCE_DIR}/resources/NucleiDetectionFilterPlugin.qrc)
  add_library(NucleiDetectionFilterPlugin SHARED ${NucleiDetection_SOURCE} ${NUCLEIDETECTION_RESOURCE})
endif(BUILD_IMAGEPROCESSING)

add_library(ColorDeconvolutionFilterPlugin SHARED ${ColorDeconvolutionPlugin_SOURCE} ${COLORDECONVOLUTION_RESOURCE})  
add_library(FilterWorkstationExtensionPlugin SHARED ${FilterWorkstationExtensionPlugin_SOURCE} ${FILTERWORKSTATIONEXTENSIONPLUGIN_RESOURCE})

if(BUILD_IMAGEPROCESSING)
	target_link_libraries(NucleiDetectionFilterPlugin PRIVATE ${OpenCV_LIBS} ASAPLib FRST basicfilters Qt5::Core Qt5::Widgets Qt5::UiTools)
    set_target_properties(NucleiDetectionFilterPlugin PROPERTIES DEBUG_POSTFIX _d)
	install(TARGETS  NucleiDetectionFilterPlugin
	  RUNTIME DESTINATION bin/plugins/filters
	  LIBRARY DESTINATION bin/plugins/filters
	  ARCHIVE DESTINATION lib/plugins/filters
	)
	install(FILES NucleiDetectionFilterPlugin.h DESTINATION include/ASAP/plugins/filters)
	if(WIN32)
		set_target_properties(NucleiDetectionFilterPlugin PROPERTIES FOLDER ASAP/plugins/filters)
	endif(WIN32)
endif(BUILD_IMAGEPROCESSING)

target_link_libraries(ColorDeconvolutionFilterPlugin PRIVATE ASAPLib basicfilters Qt5::Core Qt5::Widgets Qt5::UiTools)
target_link_libraries(FilterWorkstationExtensionPlugin PRIVATE ASAPLib basicfilters multiresolutionimageinterface Qt5::Core Qt5::Widgets Qt5::UiTools)
set_target_properties(ColorDeconvolutionFilterPlugin PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(FilterWorkstationExtensionPlugin PROPERTIES DEBUG_POSTFIX _d)

install(TARGETS ColorDeconvolutionFilterPlugin
        RUNTIME DESTINATION bin/plugins/filters
        LIBRARY DESTINATION bin/plugins/filters
        ARCHIVE DESTINATION lib/plugins/filters
)

install(TARGETS FilterWorkstationExtensionPlugin
        RUNTIME DESTINATION bin/plugins/workstationextension
        LIBRARY DESTINATION bin/plugins/workstationextension
        ARCHIVE DESTINATION lib/plugins/workstationextension
)

install(FILES ColorDeconvolutionFilterPlugin.h DESTINATION include/ASAP/plugins/filters)
install(FILES FilterThread.h FilterDockWidget.h DESTINATION include/ASAP/plugins/workstationextension)

IF(WIN32)
  set_target_properties(ColorDeconvolutionFilterPlugin PROPERTIES FOLDER ASAP/plugins/filters)
  set_target_properties(FilterWorkstationExtensionPlugin PROPERTIES FOLDER ASAP/plugins/workstationextensions)
ENDIF(WIN32)