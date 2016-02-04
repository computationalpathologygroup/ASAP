PROJECT(MultiResImageConverter)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${MultiResImageConverter_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

ADD_DEFINITIONS(${BOOST_DEFINITIONS})

SET(MultiResImageConverter_src
  MultiResImageConverter.cpp
)

IF(WIN32)
  INCLUDE_DIRECTORIES(
    ${MultiResImageConverter_SOURCE_DIR}/../../3rd-party-tk/openslide/64bit/include/openslide
  )
  LINK_DIRECTORIES(${MultiResImageConverter_SOURCE_DIR}/../../3rd-party-tk/openslide/64bit/lib)
ELSE()
  INCLUDE_DIRECTORIES(
    ${MultiResImageConverter_SOURCE_DIR}/../../3rd-party-tk/openslide/linux/include/openslide
  )
  LINK_DIRECTORIES(${MultiResImageConverter_SOURCE_DIR}/../../3rd-party-tk/openslide/linux/lib)
ENDIF()

ADD_EXECUTABLE(MultiResImageConverter ${MultiResImageConverter_src})

SET_TARGET_PROPERTIES(MultiResImageConverter PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(MultiResImageConverter multiresolutionimageinterface ${Boost_LIBRARIES})

INSTALL(
  TARGETS MultiResImageConverter
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(MultiResImageConverter PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
