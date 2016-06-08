PROJECT(Tests)

  INCLUDE_DIRECTORIES(
    .
    ../../
    ${UNITTEST_INCLUDE_DIR}
	${DIAGPathology_BINARY_DIR}
    ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
    ${CORE_BINARY_DIR}
	${BASICFILTERS_BINARY_DIR}
	${NUCLEISEGMENTATION_BINARY_DIR}
  )

  # Add test which are always performed
  FILE(GLOB unittest_io_src ${Tests_SOURCE_DIR}/../../io/*/unittest/*.cpp)
  FOREACH(TESTFILE ${unittest_io_src})
    get_filename_component(incldir ${TESTFILE} PATH)
    INCLUDE_DIRECTORIES(${incldir}/..)
  ENDFOREACH()

  SET(unittest_src
    TestRunner.cpp 
    TestData.cpp
    TestData.h
    ${unittest_io_src}
  )
  
  # Potentially add ImageProcessing tests
  IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
    FILE(GLOB unittest_imgproc_src ${Tests_SOURCE_DIR}/../../imgproc/*/unittest/*.cpp)
    FOREACH(TESTFILE ${unittest_imgproc_src})
      get_filename_component(incldir ${TESTFILE} PATH)
      INCLUDE_DIRECTORIES(${incldir}/..)
    ENDFOREACH()

    LIST(APPEND unittest_src ${unittest_imgproc_src})
  ENDIF()

  ADD_EXECUTABLE(patholTestRunner ${unittest_src})
  TARGET_LINK_LIBRARIES(patholTestRunner PRIVATE optimized ${UNITTEST_LIBRARY} debug ${UNITTEST_LIBRARY_DEBUG} multiresolutionimageinterface)
  IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
    TARGET_LINK_LIBRARIES(patholTestRunner PRIVATE basicfilters NucleiSegmentation ${OpenCV_LIBS})
  ENDIF()

  # set target properties
  SET_TARGET_PROPERTIES(patholTestRunner PROPERTIES DEBUG_POSTFIX _d)

  # set install targets
  INSTALL(TARGETS patholTestRunner
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

IF(WIN32)
  SET_TARGET_PROPERTIES(patholTestRunner PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)