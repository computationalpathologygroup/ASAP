PROJECT(BASICFILTERS)

INCLUDE_DIRECTORIES(
  ${MULTIRESOLUTIONIMAGEINTERFACE_SOURCE_DIR}
  ../../
  ${Boost_INCLUDE_DIRS}
  ${SUPERPIXELCLASSSIFICATION_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_BASICFILTERS)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(BASICFILTERS_SRCS 
  FilterBase.h
  FilterBase.cpp
  ImageFilter.h
  ImageFilter.cpp
  ColorDeconvolutionFilter.h
  ColorDeconvolutionFilter.cpp
)

ADD_LIBRARY(basicfilters SHARED ${BASICFILTERS_SRCS})
TARGET_LINK_LIBRARIES(basicfilters multiresolutionimageinterface core)

SET_TARGET_PROPERTIES(basicfilters PROPERTIES DEBUG_POSTFIX _d)
IF(WIN32)  
  SET_TARGET_PROPERTIES(basicfilters PROPERTIES FOLDER pathology/diag/imgproc)    
ENDIF(WIN32)


INSTALL(FILES FilterBase.h ColorDeconvolutionFilter.h ImageFilter.h DESTINATION include/basicfilters)

INSTALL(TARGETS basicfilters
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)