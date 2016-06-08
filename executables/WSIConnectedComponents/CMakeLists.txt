PROJECT(WSIConnectedComponents)

SET(WSIConnectedComponents_src
  WSIConnectedComponents.cpp
)

ADD_EXECUTABLE(WSIConnectedComponents ${WSIConnectedComponents_src})
SET_TARGET_PROPERTIES(WSIConnectedComponents PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIConnectedComponents wholeslidefilters core multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
TARGET_INCLUDE_DIRECTORIES(WSIConnectedComponents PRIVATE
  .
  ../../
  ${WSIConnectedComponents_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${WHOLESLIDEFILTERS_BINARY_DIR}
)
TARGET_COMPILE_DEFINITIONS(WSIConnectedComponents PRIVATE -DBOOST_ALL_DYN_LINK)

INSTALL(
  TARGETS WSIConnectedComponents
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIConnectedComponents PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
