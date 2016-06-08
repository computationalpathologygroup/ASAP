PROJECT(WSIArithmetic)

SET(WSIArithmetic_src
  WSIArithmetic.cpp
)
ADD_EXECUTABLE(WSIArithmetic ${WSIArithmetic_src})

SET_TARGET_PROPERTIES(WSIArithmetic PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIArithmetic PRIVATE wholeslidefilters core multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
TARGET_INCLUDE_DIRECTORIES(WSIArithmetic PRIVATE
  .
  ../../
  ${Boost_INCLUDE_DIRS}
  ${WSIArithmetic_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${WHOLESLIDEFILTERS_BINARY_DIR}
)
TARGET_COMPILE_DEFINITIONS(WSIArithmetic PRIVATE -DBOOST_ALL_DYN_LINK)

INSTALL(
  TARGETS WSIArithmetic 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIArithmetic PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
