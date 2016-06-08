PROJECT(OPENCVFILTERS)

FIND_PACKAGE(OpenCV REQUIRED)

SET(OPENCVFILTERS_SRCS 
  DIAGPathologyOpenCVBridge.h
  NucleiDetectionFilter.h
  NucleiDetectionFilter.cpp
)

ADD_LIBRARY(opencvfilters SHARED ${OPENCVFILTERS_SRCS})
TARGET_INCLUDE_DIRECTORIES(
  opencvfilters
  PRIVATE
  ${DIAGPathology_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)

IF(OpenCV_VERSION EQUAL "3.1.0")
TARGET_LINK_LIBRARIES(opencvfilters PRIVATE core optimized ${OpenCV_LIB_DIR_OPT}/opencv_world310.lib debug ${OpenCV_LIB_DIR_DBG}/opencv_world310d.lib)
ELSE()
TARGET_LINK_LIBRARIES(opencvfilters PRIVATE core ${OpenCV_LIBS})
ENDIF()

IF(WIN32)  
  SET_TARGET_PROPERTIES(opencvfilters PROPERTIES FOLDER pathology/diag/imgproc)
ENDIF(WIN32)


INSTALL(FILES DIAGPathologyOpenCVBridge.h NucleiDetectionFilter.h DESTINATION include/imgproc/opencvfilters)

INSTALL(TARGETS opencvfilters
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)