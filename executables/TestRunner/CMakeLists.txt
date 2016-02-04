PROJECT(Tests)

  INCLUDE_DIRECTORIES(
    .
    ../../
    ${Boost_INCLUDE_DIRS}
    ${UNITTEST_INCLUDE_DIR}
	${DIAGPathology_BINARY_DIR}
  )
  ADD_DEFINITIONS(${Boost_DEFINITIONS})
  ADD_DEFINITIONS(-DNOMINMAX)
  LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

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

  IF(WIN32)
    LINK_DIRECTORIES(${Tests_SOURCE_DIR}/../../3rd-party-tk/openslide/64bit/lib)
  ELSE()
    LINK_DIRECTORIES(${Tests_SOURCE_DIR}/../../3rd-party-tk/openslide/linux/lib)
  ENDIF()

  ADD_EXECUTABLE(patholTestRunner ${unittest_src})
  TARGET_LINK_LIBRARIES(patholTestRunner optimized ${UNITTEST_LIBRARY} debug ${UNITTEST_LIBRARY_DEBUG} multiresolutionimageinterface)
  IF(BUILD_DIAG_PATHOLOGY_IMAGEPROCESSING)
    TARGET_LINK_LIBRARIES(patholTestRunner basicfilters NucleiSegmentation ${OpenCV_LIBS})
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