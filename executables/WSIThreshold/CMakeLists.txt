PROJECT(WSIThreshold)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${WSIThreshold_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(WSIThreshold_src
  WSIThreshold.cpp
)

ADD_EXECUTABLE(WSIThreshold ${WSIThreshold_src})

SET_TARGET_PROPERTIES(WSIThreshold PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIThreshold wholeslidefilters multiresolutionimageinterface)

INSTALL(
  TARGETS WSIThreshold 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIThreshold  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
