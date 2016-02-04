PROJECT(WSIArithmetic)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${WSIArithmetic_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(WSIArithmetic_src
  WSIArithmetic.cpp
)

ADD_EXECUTABLE(WSIArithmetic ${WSIArithmetic_src})

SET_TARGET_PROPERTIES(WSIArithmetic PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIArithmetic wholeslidefilters multiresolutionimageinterface)

INSTALL(
  TARGETS WSIArithmetic 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIArithmetic  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
