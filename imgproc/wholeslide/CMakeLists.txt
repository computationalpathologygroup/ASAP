set(WHOLESLIDEFILTERS_SRCS 
    ConnectedComponentsWholeSlideFilter.h
    ConnectedComponentsWholeSlideFilter.cpp
    DistanceTransformWholeSlideFilter.h
    DistanceTransformWholeSlideFilter.cpp
    LabelStatisticsWholeSlideFilter.h
    LabelStatisticsWholeSlideFilter.cpp
    ThresholdWholeSlideFilter.h
    ThresholdWholeSlideFilter.cpp
    ArithmeticWholeSlideFilter.h
    ArithmeticWholeSlideFilter.cpp
    NucleiDetectionWholeSlideFilter.h
    NucleiDetectionWholeSlideFilter.cpp
)

add_library(wholeslidefilters SHARED ${WHOLESLIDEFILTERS_SRCS})
target_link_libraries(wholeslidefilters PRIVATE ${OpenCV_LIBS} annotation FRST basicfilters multiresolutionimageinterface)
target_include_directories(wholeslidefilters PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/imgproc/wholeslidefilters>)
generate_export_header(wholeslidefilters)

set_target_properties(wholeslidefilters PROPERTIES DEBUG_POSTFIX _d)
IF(WIN32)  
  set_target_properties(wholeslidefilters PROPERTIES FOLDER imgproc)    
ENDIF(WIN32)

install(FILES ConnectedComponentsWholeSlideFilter.h DistanceTransformWholeSlideFilter.h LabelStatisticsWholeSlideFilter.h ThresholdWholeSlideFilter.h ArithmeticWholeSlideFilter.h DESTINATION include/imgproc/wholeslidefilters)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/wholeslidefilters_export.h DESTINATION include/imgproc/wholeslidefilters)

install(TARGETS wholeslidefilters
        EXPORT asap-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

OPTION(WRAP_WHOLESLIDEFILTERS_PYTHON "Wrap whole slide image processing in Python" OFF)

IF(WRAP_WHOLESLIDEFILTERS_PYTHON)
  add_definitions(-DSWIG_PYTHON_INTERPRETER_NO_DEBUG)
  find_package(PythonLibs REQUIRED)
  find_package(SWIG REQUIRED)
  include (${SWIG_USE_FILE})
  if(SWIG_FOUND)
    if(NOT WIN32)
      set(CMAKE_SWIG_FLAGS "-I${Boost_INCLUDE_DIR}")
    endif(NOT WIN32)     
    mark_as_advanced(SWIG_DIR SWIG_VERSION)
  endif(SWIG_FOUND)

	# Wrapping part for Python
	IF (WRAP_WHOLESLIDEFILTERS_PYTHON AND SWIG_FOUND)   
	  # Set module extension, platform specific
	  IF(WIN32)
		SET(module_extension ".pyd")
	  ELSE(WIN32)
		SET(module_extension ".so")  
	  ENDIF(WIN32)
	  
	  set_source_files_properties(wholeslidefilters.i PROPERTIES CPLUSPLUS ON)
	  set_source_files_properties(wholeslidefilters.i PROPERTIES SWIG_FLAGS "-builtin")
	  swig_add_library(wholeslidefilters LANGUAGE python SOURCES wholeslidefilters.i)
	  swig_link_libraries(wholeslidefilters wholeslidefilters core) 
	  swig_link_libraries(wholeslidefilters ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES})
	  target_include_directories(${SWIG_MODULE_wholeslidefilters_REAL_NAME} PRIVATE ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
	  
	  set_target_properties(${SWIG_MODULE_wholeslidefilters_REAL_NAME} PROPERTIES FOLDER imgproc)
	  set_target_properties(${SWIG_MODULE_wholeslidefilters_REAL_NAME} PROPERTIES DEBUG_POSTFIX _d)
	  
	  # Cross-platform copying and installing instructions
	  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/wholeslidefilters.py DESTINATION bin)

      INSTALL(TARGETS ${SWIG_MODULE_wholeslidefilters_REAL_NAME}
	  	      EXPORT asap-targets
              RUNTIME DESTINATION bin
              LIBRARY DESTINATION bin
              ARCHIVE DESTINATION lib
      )
	ENDIF(WRAP_WHOLESLIDEFILTERS_PYTHON AND SWIG_FOUND)  
ENDIF(WRAP_WHOLESLIDEFILTERS_PYTHON)