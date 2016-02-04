PROJECT(WSILabelStatistics)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${WSILabelStatistics_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(WSILabelStatistics_src
  WSILabelStatistics.cpp
)

ADD_EXECUTABLE(WSILabelStatistics ${WSILabelStatistics_src})

SET_TARGET_PROPERTIES(WSILabelStatistics PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSILabelStatistics wholeslidefilters multiresolutionimageinterface)

INSTALL(
  TARGETS WSILabelStatistics 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSILabelStatistics  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
