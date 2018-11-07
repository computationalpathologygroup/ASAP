# Add test which are always performed
file(GLOB unittest_io_src ${CMAKE_CURRENT_SOURCE_DIR}/../../multiresolutionimageinterface/unittest/*.cpp)
foreach(TESTFILE ${unittest_io_src})
  get_filename_component(incldir ${TESTFILE} PATH)
  include_directories(${incldir}/..)
ENDFOREACH()

set(unittest_src
    TestRunner.cpp 
    TestData.cpp
    TestData.h
    ${unittest_io_src}
)
  
# Potentially add ImageProcessing tests
if(BUILD_IMAGEPROCESSING)
  file(GLOB unittest_imgproc_src ${CMAKE_CURRENT_SOURCE_DIR}/../../imgproc/*/unittest/*.cpp)
  foreach(TESTFILE ${unittest_imgproc_src})
    get_filename_component(incldir ${TESTFILE} PATH)
    include_directories(${incldir}/..)
  endforeach()

  list(APPEND unittest_src ${unittest_imgproc_src})
endif()

add_executable(testRunner ${unittest_src})
target_include_directories(testRunner PRIVATE ${UNITTEST_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(testRunner PRIVATE optimized ${UNITTEST_LIBRARY} debug ${UNITTEST_LIBRARY_DEBUG} multiresolutionimageinterface)
if(BUILD_IMAGEPROCESSING)
  target_link_libraries(testRunner PRIVATE basicfilters FRST ${OpenCV_LIBS})
endif()

# set target properties
set_target_properties(testRunner PROPERTIES DEBUG_POSTFIX _d)

# set install targets
install(TARGETS testRunner
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(testRunner PROPERTIES FOLDER executables)   
endif(WIN32)