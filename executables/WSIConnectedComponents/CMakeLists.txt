PROJECT(WSIConnectedComponents)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${WSIConnectedComponents_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(WSIConnectedComponents_src
  WSIConnectedComponents.cpp
)

ADD_EXECUTABLE(WSIConnectedComponents ${WSIConnectedComponents_src})

SET_TARGET_PROPERTIES(WSIConnectedComponents PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIConnectedComponents wholeslidefilters multiresolutionimageinterface)

INSTALL(
  TARGETS WSIConnectedComponents
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIConnectedComponents PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
