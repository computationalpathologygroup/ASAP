PROJECT(WSIDistanceTransform)

SET(WSIDistanceTransform_src
  WSIDistanceTransform.cpp
)

ADD_EXECUTABLE(WSIDistanceTransform ${WSIDistanceTransform_src})
SET_TARGET_PROPERTIES(WSIDistanceTransform PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(WSIDistanceTransform wholeslidefilters multiresolutionimageinterface core Boost::disable_autolinking Boost::program_options)
TARGET_INCLUDE_DIRECTORIES(
  WSIDistanceTransform PRIVATE
  .
  ../../
  ${WSIDistanceTransform_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
  ${WHOLESLIDEFILTERS_BINARY_DIR}
)
TARGET_COMPILE_DEFINITIONS(WSIDistanceTransform PRIVATE -DBOOST_ALL_DYN_LINK)


INSTALL(
  TARGETS WSIDistanceTransform 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(WSIDistanceTransform  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
