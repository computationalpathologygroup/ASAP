PROJECT(WSIDistanceTransform)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${WSIDistanceTransform_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(WSIDistanceTransform_src
  WSIDistanceTransform.cpp
)

ADD_EXECUTABLE(WSIDistanceTransform ${WSIDistanceTransform_src})

SET_TARGET_PROPERTIES(WSIDistanceTransform PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIDistanceTransform wholeslidefilters multiresolutionimageinterface)

INSTALL(
  TARGETS WSIDistanceTransform 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIDistanceTransform  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
