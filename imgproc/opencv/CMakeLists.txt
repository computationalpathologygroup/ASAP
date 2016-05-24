PROJECT(OPENCVFILTERS)

IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
  FIND_PACKAGE(OpenCV REQUIRED)
ENDIF()

INCLUDE_DIRECTORIES(
  ${DIAGPathology_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${Boost_INCLUDE_DIRS}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_OPENCVFILTERS)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(OPENCVFILTERS_SRCS 
  DIAGPathologyOpenCVBridge.h
  NucleiDetectionFilter.h
  NucleiDetectionFilter.cpp
)

ADD_LIBRARY(opencvfilters SHARED ${OPENCVFILTERS_SRCS})
TARGET_LINK_LIBRARIES(opencvfilters core)
IF(OpenCV_VERSION EQUAL "3.1.0")
TARGET_LINK_LIBRARIES(opencvfilters optimized ${OpenCV_LIB_DIR_OPT}/opencv_world310.lib debug ${OpenCV_LIB_DIR_DBG}/opencv_world310d.lib)
ELSE()
TARGET_LINK_LIBRARIES(opencvfilters ${OpenCV_LIBS})
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