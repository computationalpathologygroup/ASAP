PROJECT(BASICFILTERS)

SET(BASICFILTERS_SRCS 
  FilterBase.h
  FilterBase.cpp
  ImageFilter.h
  ImageFilter.cpp
  ColorDeconvolutionFilter.h
  ColorDeconvolutionFilter.cpp
)

ADD_LIBRARY(basicfilters SHARED ${BASICFILTERS_SRCS})
TARGET_INCLUDE_DIRECTORIES(
  basicfilters
  PRIVATE
  ${MULTIRESOLUTIONIMAGEINTERFACE_SOURCE_DIR}
  ../../
  ${DIAGPathology_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
)

TARGET_LINK_LIBRARIES(basicfilters multiresolutionimageinterface core)
generate_export_header(basicfilters)

SET_TARGET_PROPERTIES(basicfilters PROPERTIES DEBUG_POSTFIX _d)
IF(WIN32)  
  SET_TARGET_PROPERTIES(basicfilters PROPERTIES FOLDER pathology/diag/imgproc)    
ENDIF(WIN32)


INSTALL(FILES FilterBase.h ColorDeconvolutionFilter.h ImageFilter.h DESTINATION include/imgproc/basicfilters)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/basicfilters_export.h DESTINATION include/imgproc/basicfilters)

INSTALL(TARGETS basicfilters
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)