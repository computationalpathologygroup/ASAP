PROJECT(WSIThreshold)

SET(WSIThreshold_src
  WSIThreshold.cpp
)

ADD_EXECUTABLE(WSIThreshold ${WSIThreshold_src})
SET_TARGET_PROPERTIES(WSIThreshold PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIThreshold wholeslidefilters multiresolutionimageinterface core Boost::disable_autolinking Boost::program_options)
TARGET_INCLUDE_DIRECTORIES(
  WSIThreshold  
  PRIVATE
  .
  ../../
  ${WSIThreshold_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${WHOLESLIDEFILTERS_BINARY_DIR}
)
TARGET_COMPILE_DEFINITIONS(WSIThreshold PRIVATE -DBOOST_ALL_DYN_LINK)

INSTALL(
  TARGETS WSIThreshold 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIThreshold  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
