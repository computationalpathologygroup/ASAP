PROJECT(WHOLESLIDEFILTERS)

INCLUDE_DIRECTORIES(
  ${MULTIRESOLUTIONIMAGEINTERFACE_SOURCE_DIR}
  ../../
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_WHOLESLIDEFILTERS)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(WHOLESLIDEFILTERS_SRCS 
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

ADD_LIBRARY(wholeslidefilters SHARED ${WHOLESLIDEFILTERS_SRCS})
TARGET_LINK_LIBRARIES(wholeslidefilters ${OpenCV_LIBS} annotation NucleiSegmentation basicfilters multiresolutionimageinterface core)

SET_TARGET_PROPERTIES(wholeslidefilters PROPERTIES DEBUG_POSTFIX _d)
IF(WIN32)  
  SET_TARGET_PROPERTIES(wholeslidefilters PROPERTIES FOLDER pathology/diag/imgproc)    
ENDIF(WIN32)

INSTALL(FILES ConnectedComponentsWholeSlideFilter.h DistanceTransformWholeSlideFilter.h LabelStatisticsWholeSlideFilter.h ThresholdWholeSlideFilter.h ArithmeticWholeSlideFilter.h DESTINATION include/wholeslidefilters)

INSTALL(TARGETS wholeslidefilters
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
  IF(SWIG_FOUND)
    if(NOT WIN32)
      set(CMAKE_SWIG_FLAGS "-I${Boost_INCLUDE_DIR}")
    endif(NOT WIN32)     
    MARK_AS_ADVANCED(SWIG_DIR SWIG_VERSION)
  ENDIF(SWIG_FOUND)

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
	  include_directories(${PYTHON_INCLUDE_DIR})
	  IF(PYTHON_CONFIG_INCLUDE_DIR)
		include_directories(${PYTHON_CONFIG_INCLUDE_DIR})
	  ENDIF(PYTHON_CONFIG_INCLUDE_DIR)

	  SWIG_ADD_MODULE(wholeslidefilters python wholeslidefilters.i)
	  SWIG_LINK_LIBRARIES(wholeslidefilters wholeslidefilters core) 
	  SWIG_LINK_LIBRARIES(wholeslidefilters ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES})
	  SET_TARGET_PROPERTIES(${SWIG_MODULE_wholeslidefilters_REAL_NAME} PROPERTIES FOLDER pathology/diag/imgproc)
	  SET_TARGET_PROPERTIES(${SWIG_MODULE_wholeslidefilters_REAL_NAME} PROPERTIES DEBUG_POSTFIX _d)
	  
	  # Cross-platform copying and installing instructions
	  INSTALL(FILES ${PROJECT_BINARY_DIR}/wholeslidefilters.py DESTINATION bin)

      INSTALL(TARGETS ${SWIG_MODULE_wholeslidefilters_REAL_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
      )
	ENDIF(WRAP_WHOLESLIDEFILTERS_PYTHON AND SWIG_FOUND)  
ENDIF(WRAP_WHOLESLIDEFILTERS_PYTHON)