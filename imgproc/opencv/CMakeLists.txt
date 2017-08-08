PROJECT(OPENCVFILTERS)

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

TARGET_LINK_LIBRARIES(opencvfilters PRIVATE core ${OpenCV_LIBS})

IF(WIN32)  
  SET_TARGET_PROPERTIES(opencvfilters PROPERTIES FOLDER pathology/diag/imgproc)
ENDIF(WIN32)


INSTALL(FILES DIAGPathologyOpenCVBridge.h NucleiDetectionFilter.h DESTINATION include/imgproc/opencvfilters)

INSTALL(TARGETS opencvfilters
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)