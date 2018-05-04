PROJECT(FilterPlugins)

INCLUDE_DIRECTORIES(
../
../../
${DIAGPathology_BINARY_DIR}
${CORE_BINARY_DIR}
${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
${BASICFILTERS_BINARY_DIR}
${NUCLEISEGMENTATION_BINARY_DIR}
${CORE_BINARY_DIR}
${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
${BASICFILTERS_BINARY_DIR}
)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET(FilterWorkstationExtensionPlugin_SOURCE
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

ADD_DEFINITIONS(-DBUILD_FILTEREXTENSIONNPLUGIN)


find_package(Qt5 COMPONENTS Core Widgets Gui)  
qt5_add_resources(COLORDECONVOLUTION_RESOURCE ${FilterPlugins_SOURCE_DIR}/resources/ColorDeconvolutionFilterPlugin.qrc)
qt5_add_resources(FILTERWORKSTATIONEXTENSIONPLUGIN_RESOURCE ${FilterPlugins_SOURCE_DIR}/resources/FilterWorkstationExtensionPlugin_resources.qrc)

IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
qt5_add_resources(NUCLEIDETECTION_RESOURCE ${FilterPlugins_SOURCE_DIR}/resources/NucleiDetectionFilterPlugin.qrc)
ADD_LIBRARY(NucleiDetectionFilterPlugin SHARED ${NucleiDetection_SOURCE} ${NUCLEIDETECTION_RESOURCE})
QT5_USE_MODULES(NucleiDetectionFilterPlugin Core Widgets UiTools)  
ENDIF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)

ADD_LIBRARY(ColorDeconvolutionFilterPlugin SHARED ${ColorDeconvolutionPlugin_SOURCE} ${COLORDECONVOLUTION_RESOURCE})  
QT5_USE_MODULES(ColorDeconvolutionFilterPlugin Core Widgets UiTools)
add_library(FilterWorkstationExtensionPlugin SHARED ${FilterWorkstationExtensionPlugin_SOURCE} ${FILTERWORKSTATIONEXTENSIONPLUGIN_RESOURCE})
qt5_use_modules(FilterWorkstationExtensionPlugin Core Widgets UiTools)

IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
	TARGET_LINK_LIBRARIES(NucleiDetectionFilterPlugin ${OpenCV_LIBS} NucleiSegmentation basicfilters multiresolutionimageinterface core)
    SET_TARGET_PROPERTIES(NucleiDetectionFilterPlugin PROPERTIES DEBUG_POSTFIX _d)
	INSTALL(TARGETS  NucleiDetectionFilterPlugin
	  RUNTIME DESTINATION bin/plugins/filters
	  LIBRARY DESTINATION bin/plugins/filters
	  ARCHIVE DESTINATION lib/plugins/filters
	)
	INSTALL(FILES NucleiDetectionFilterPlugin.h DESTINATION include/ASAP/plugins/filters)
	IF(WIN32)
		SET_TARGET_PROPERTIES(NucleiDetectionFilterPlugin PROPERTIES FOLDER pathology/diag/ASAP/plugins/filters)
	ENDIF(WIN32)
ENDIF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)

TARGET_LINK_LIBRARIES(ColorDeconvolutionFilterPlugin basicfilters multiresolutionimageinterface core)
TARGET_LINK_LIBRARIES(FilterWorkstationExtensionPlugin ASAPLib basicfilters multiresolutionimageinterface core)
SET_TARGET_PROPERTIES(ColorDeconvolutionFilterPlugin PROPERTIES DEBUG_POSTFIX _d)
SET_TARGET_PROPERTIES(FilterWorkstationExtensionPlugin PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS ColorDeconvolutionFilterPlugin
  RUNTIME DESTINATION bin/plugins/filters
  LIBRARY DESTINATION bin/plugins/filters
  ARCHIVE DESTINATION lib/plugins/filters
)

INSTALL(TARGETS FilterWorkstationExtensionPlugin
  RUNTIME DESTINATION bin/plugins/workstationextension
  LIBRARY DESTINATION bin/plugins/workstationextension
  ARCHIVE DESTINATION lib/plugins/workstationextension
)

INSTALL(FILES ColorDeconvolutionFilterPlugin.h DESTINATION include/ASAP/plugins/filters)
INSTALL(FILES FilterThread.h FilterDockWidget.h DESTINATION include/ASAP/plugins/workstationextension)

IF(WIN32)
  SET_TARGET_PROPERTIES(ColorDeconvolutionFilterPlugin PROPERTIES FOLDER pathology/diag/ASAP/plugins/filters)
  SET_TARGET_PROPERTIES(FilterWorkstationExtensionPlugin PROPERTIES FOLDER pathology/diag/ASAP/plugins/workstationextensions)
ENDIF(WIN32)