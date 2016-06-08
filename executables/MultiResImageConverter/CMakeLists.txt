PROJECT(MultiResImageConverter)

SET(MultiResImageConverter_src
  MultiResImageConverter.cpp
)

ADD_EXECUTABLE(MultiResImageConverter ${MultiResImageConverter_src})
SET_TARGET_PROPERTIES(MultiResImageConverter PROPERTIES DEBUG_POSTFIX _d)
TARGET_INCLUDE_DIRECTORIES(
  MultiResImageConverter PRIVATE
  .
  ../../
  ${Boost_INCLUDE_DIRS}
  ${MultiResImageConverter_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${DIAGPathology_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${CORE_BINARY_DIR}
)
TARGET_COMPILE_DEFINITIONS(MultiResImageConverter PRIVATE -DBOOST_ALL_DYN_LINK)
TARGET_LINK_LIBRARIES(MultiResImageConverter PRIVATE multiresolutionimageinterface core Boost::disable_autolinking Boost::program_options)

INSTALL(
  TARGETS MultiResImageConverter
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(MultiResImageConverter PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
