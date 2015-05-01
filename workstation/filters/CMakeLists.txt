PROJECT(WorkstationPlugins)

IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
  FIND_PACKAGE(OpenCV REQUIRED)
ENDIF()

INCLUDE_DIRECTORIES(
../
../../
${SUPERPIXELCLASSSIFICATION_SOURCE_DIR}
)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(ColorDeconvolutionPlugin_SOURCE 
  ColorDeconvolutionFilterPlugin.h
  ColorDeconvolutionFilterPlugin.cpp
)

set(NucleiDetection_SOURCE 
  NucleiDetectionFilterPlugin.h
  NucleiDetectionFilterPlugin.cpp
)

IF(BUILD_USING_QT4)
  QT4_WRAP_CPP(NucleiDetectionPlugin_MOC NucleiDetectionFilterPlugin.h)
  QT4_WRAP_CPP(ColorDeconvolutionPlugin_MOC ColorDeconvolutionFilterPlugin.h)
  ADD_LIBRARY(NucleiDetectionFilterPlugin SHARED ${NucleiDetection_SOURCE} ${NucleiDetectionPlugin_MOC})
  ADD_LIBRARY(ColorDeconvolutionFilterPlugin SHARED ${ColorDeconvolutionPlugin_SOURCE} ${ColorDeconvolutionPlugin_MOC})
  TARGET_LINK_LIBRARIES(NucleiDetectionFilterPlugin ${QT_LIBRARIES} )
  TARGET_LINK_LIBRARIES(ColorDeconvolutionFilterPlugin ${QT_LIBRARIES} )  
ELSE()
  find_package(Qt5 COMPONENTS Core Widgets Gui)  
  qt5_add_resources(COLORDECONVOLUTION_RESOURCE ColorDeconvolutionFilterPlugin.qrc)
  qt5_add_resources(NUCLEIDETECTION_RESOURCE NucleiDetectionFilterPlugin.qrc)
  ADD_LIBRARY(NucleiDetectionFilterPlugin SHARED ${NucleiDetection_SOURCE} ${NUCLEIDETECTION_RESOURCE})
  QT5_USE_MODULES(NucleiDetectionFilterPlugin Core Widgets UiTools)  
  ADD_LIBRARY(ColorDeconvolutionFilterPlugin SHARED ${ColorDeconvolutionPlugin_SOURCE} ${COLORDECONVOLUTION_RESOURCE})  
  QT5_USE_MODULES(ColorDeconvolutionFilterPlugin Core Widgets UiTools)
ENDIF()

TARGET_LINK_LIBRARIES(NucleiDetectionFilterPlugin ${OpenCV_LIBS} NucleiSegmentation basicfilters multiresolutionimageinterface core)
TARGET_LINK_LIBRARIES(ColorDeconvolutionFilterPlugin basicfilters multiresolutionimageinterface core)
SET_TARGET_PROPERTIES(NucleiDetectionFilterPlugin PROPERTIES DEBUG_POSTFIX _d)
SET_TARGET_PROPERTIES(ColorDeconvolutionFilterPlugin PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS ColorDeconvolutionFilterPlugin NucleiDetectionFilterPlugin
  RUNTIME DESTINATION bin/plugins/filters
  LIBRARY DESTINATION lib/plugins/filters
  ARCHIVE DESTINATION lib/plugins/filters
)

INSTALL(FILES NucleiDetectionFilterPlugin.h ColorDeconvolutionFilterPlugin.h DESTINATION include/workstation/plugins/filters)

IF(WIN32)
  SET_TARGET_PROPERTIES(NucleiDetectionFilterPlugin PROPERTIES FOLDER pathology/diag/workstation/plugins/filters)
  SET_TARGET_PROPERTIES(ColorDeconvolutionFilterPlugin PROPERTIES FOLDER pathology/diag/workstation/plugins/filters)
ENDIF(WIN32)