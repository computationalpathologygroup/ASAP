PROJECT(MULTIRESOLUTIONIMAGEINTERFACE)

INCLUDE_DIRECTORIES(
  ${MULTIRESOLUTIONIMAGEINTERFACE_SOURCE_DIR} 
  ../core
  ../../
  ../
  ${Boost_INCLUDE_DIRS}
  ${TIFF_INCLUDE_DIR}
  ${DCMTKJPEG_INCLUDE_DIR}
  ${JASPER_INCLUDE_DIR}
  ${OPENSLIDE_INCLUDE_DIR}
  ${PugiXML_INCLUDE_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

if(WIN32 AND NOT MINGW)
  add_definitions(-DJAS_WIN_MSVC_BUILD)
  add_definitions(-DNOMINMAX)
endif(WIN32 AND NOT MINGW)

IF(WIN32)
  ADD_DEFINITIONS(-DBUILD_MULTIRESOLUTIONIMAGEINTERFACE)
ENDIF(WIN32)

SET(MULTIRESOLUTIONIMAGEINTERFACE_HS
    JPEG2000Codec.h 
    MultiResolutionImageReader.h 
    MultiResolutionImageWriter.h 
	AperioSVSWriter.h
    VSIImage.h
    TIFFImage.h
    OpenSlideImage.h
    MultiResolutionImage.h
    TileCache.h
    LIFImage.h
)

SET(MULTIRESOLUTIONIMAGEINTERFACE_SRCS 
    JPEG2000Codec.cpp 
    MultiResolutionImageReader.cpp 
    MultiResolutionImageWriter.cpp 
	AperioSVSWriter.cpp
    VSIImage.cpp
    TIFFImage.cpp
    OpenSlideImage.cpp  
    MultiResolutionImage.cpp
    TileCache.cpp
    LIFImage.cpp
)

ADD_LIBRARY(multiresolutionimageinterface SHARED ${MULTIRESOLUTIONIMAGEINTERFACE_SRCS} ${MULTIRESOLUTIONIMAGEINTERFACE_HS})
TARGET_LINK_LIBRARIES(multiresolutionimageinterface ${DCMTKJPEG_LIBRARY} libtiff ${OPENSLIDE_LIBRARY} core libjasper ${Boost_THREAD_LIBRARY} libjpeg zlib pugixml)

SET_TARGET_PROPERTIES(multiresolutionimageinterface PROPERTIES DEBUG_POSTFIX _d)

IF(WIN32)  
  SET_TARGET_PROPERTIES(multiresolutionimageinterface PROPERTIES FOLDER pathology/diag/io)    
ENDIF(WIN32)

OPTION(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON "Wrap multi-resolution image reading in Python" OFF)

IF(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)
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
	IF (WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON AND SWIG_FOUND)   
	  # Set module extension, platform specific
	  IF(WIN32)
		SET(module_extension ".pyd")
	  ELSE(WIN32)
		SET(module_extension ".so")  
	  ENDIF(WIN32)
	  
	  set_source_files_properties(multiresolutionimageinterface.i PROPERTIES CPLUSPLUS ON)
	  set_source_files_properties(multiresolutionimageinterface.i PROPERTIES SWIG_FLAGS "-builtin")
	  include_directories(${PYTHON_INCLUDE_DIR})
	  include_directories(${PYTHON_INCLUDE_DIR}/../Lib/site-packages/numpy/core/include)
	  IF(PYTHON_CONFIG_INCLUDE_DIR)
		include_directories(${PYTHON_CONFIG_INCLUDE_DIR})
	  ENDIF(PYTHON_CONFIG_INCLUDE_DIR)

	  SWIG_ADD_MODULE(multiresolutionimageinterface python multiresolutionimageinterface.i)
	  SWIG_LINK_LIBRARIES(multiresolutionimageinterface multiresolutionimageinterface core) 
	  SWIG_LINK_LIBRARIES(multiresolutionimageinterface ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES})
	  SET_TARGET_PROPERTIES(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PROPERTIES FOLDER pathology/diag/io)
	  SET_TARGET_PROPERTIES(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PROPERTIES DEBUG_POSTFIX _d)
	  
	  # Cross-platform copying and installing instructions
	  # ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/multiresolutionimageinterface.py ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}) 
	  # ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/multiresolutionimageinterface.py ${CMAKE_CURRENT_SOURCE_DIR}/../../mevislab/DIAG/Pathology/Modules/Scripts/python/multiresolutionimageinterface/multiresolutionimageinterface.py) 
	  # ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME}>" "${CMAKE_CURRENT_SOURCE_DIR}/../../mevislab/DIAG/Pathology/Modules/Scripts/python/multiresolutionimageinterface/$<TARGET_FILE_NAME:${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME}>")
	ENDIF(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON AND SWIG_FOUND)  
ENDIF(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)

INSTALL(FILES ${MULTIRESOLUTIONIMAGEINTERFACE_HS} DESTINATION include/multiresolutionimageinterface)

INSTALL(TARGETS multiresolutionimageinterface
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)