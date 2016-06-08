PROJECT(WSILabelStatistics)

SET(WSILabelStatistics_src
  WSILabelStatistics.cpp
)

ADD_EXECUTABLE(WSILabelStatistics ${WSILabelStatistics_src})
SET_TARGET_PROPERTIES(WSILabelStatistics PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSILabelStatistics wholeslidefilters multiresolutionimageinterface core  Boost::disable_autolinking Boost::program_options)
TARGET_INCLUDE_DIRECTORIES(
  WSILabelStatistics 
  PRIVATE
  .
  ../../
  ${WSILabelStatistics_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${WHOLESLIDEFILTERS_BINARY_DIR}
)
TARGET_COMPILE_DEFINITIONS(WSILabelStatistics PRIVATE -DBOOST_ALL_DYN_LINK)


INSTALL(
  TARGETS WSILabelStatistics 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSILabelStatistics  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
