PROJECT(MULTIRESOLUTIONIMAGEINTERFACE)

OPTION(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT "Builds support for Olympus VSI files; requires DCMTK ijg8.lib" OFF)

INCLUDE_DIRECTORIES(
  ${MULTIRESOLUTIONIMAGEINTERFACE_SOURCE_DIR} 
  ../../
  ../
  ${Boost_INCLUDE_DIRS}
  ${TIFF_INCLUDE_DIR}
  ${JASPER_INCLUDE_DIR}
  ${OPENSLIDE_INCLUDE_DIR}
  ${PugiXML_INCLUDE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CORE_BINARY_DIR}
)

IF(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
INCLUDE_DIRECTORIES(
  ${DCMTKJPEG_INCLUDE_DIR}
)
ENDIF()

if(WIN32 AND NOT MINGW)
  add_definitions(-DJAS_WIN_MSVC_BUILD)
  add_definitions(-DNOMINMAX)
endif(WIN32 AND NOT MINGW)

SET(MULTIRESOLUTIONIMAGEINTERFACE_HS
    JPEG2000Codec.h 
    MultiResolutionImageReader.h 
    MultiResolutionImageWriter.h 
	AperioSVSWriter.h
    TIFFImage.h
	TIFFImageFactory.h
    MultiResolutionImage.h
	MultiResolutionImageFactory.h
    TileCache.h
    LIFImage.h
	LIFImageFactory.h
)

SET(MULTIRESOLUTIONIMAGEINTERFACE_SRCS 
    JPEG2000Codec.cpp 
    MultiResolutionImageReader.cpp 
    MultiResolutionImageWriter.cpp 
	AperioSVSWriter.cpp
    TIFFImage.cpp
    MultiResolutionImage.cpp
	MultiResolutionImageFactory.cpp
    MultiResolutionImage.cpp
	TIFFImageFactory.cpp
    TileCache.cpp
    LIFImage.cpp
	LIFImageFactory.cpp
)

SET(EXTERNAL_OPENSLIDE_HS
    OpenSlideImage.h
	OpenSlideImageFactory.h
)

SET(VSI_SOURCE_HS
    VSIImage.h
	VSIImageFactory.h
)

SET(EXTERNAL_OPENSLIDE_SRCS
    OpenSlideImage.cpp
	OpenSlideImageFactory.cpp
)

SET(VSI_SOURCE_SRCS
    VSIImage.cpp
	VSIImageFactory.cpp
)

IF(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
# Required for lossless JPEG compression used in VSIs
FIND_PACKAGE(DCMTKJPEG REQUIRED)
ADD_LIBRARY(multiresolutionimageinterface SHARED ${MULTIRESOLUTIONIMAGEINTERFACE_SRCS} ${MULTIRESOLUTIONIMAGEINTERFACE_HS} ${VSI_SOURCE_HS} ${VSI_SOURCE_SRCS})
TARGET_LINK_LIBRARIES(multiresolutionimageinterface PRIVATE ${DCMTKJPEG_LIBRARY} libtiff core libjasper libjpeg zlib pugixml Boost::disable_autolinking Boost::thread)
TARGET_COMPILE_DEFINITIONS(multiresolutionimageinterface PRIVATE HAS_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
ELSE(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
ADD_LIBRARY(multiresolutionimageinterface SHARED ${MULTIRESOLUTIONIMAGEINTERFACE_SRCS} ${MULTIRESOLUTIONIMAGEINTERFACE_HS})
TARGET_LINK_LIBRARIES(multiresolutionimageinterface PRIVATE libtiff core libjasper libjpeg zlib pugixml Boost::disable_autolinking Boost::thread)
ENDIF(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)

generate_export_header(multiresolutionimageinterface)
SET_TARGET_PROPERTIES(multiresolutionimageinterface PROPERTIES DEBUG_POSTFIX _d)

ADD_LIBRARY(openslidefileformat SHARED ${EXTERNAL_OPENSLIDE_SRCS} ${EXTERNAL_OPENSLIDE_HS})
generate_export_header(openslidefileformat)
TARGET_LINK_LIBRARIES(openslidefileformat PRIVATE ${OPENSLIDE_LIBRARY} multiresolutionimageinterface core Boost::disable_autolinking Boost::thread)
SET_TARGET_PROPERTIES(openslidefileformat PROPERTIES DEBUG_POSTFIX _d)

IF(WIN32)  
  SET_TARGET_PROPERTIES(multiresolutionimageinterface PROPERTIES FOLDER pathology/diag/io)    
  SET_TARGET_PROPERTIES(openslidefileformat PROPERTIES FOLDER pathology/diag/io)
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
      IF(PYTHONLIBS_VERSION_STRING MATCHES "3.5")
	      set_property(SOURCE multiresolutionimageinterface.i PROPERTY SWIG_FLAGS "-py3")
	  ELSEIF(PYTHONLIBS_VERSION_STRING MATCHES "3.5")
	      set_property(SOURCE multiresolutionimageinterface.i PROPERTY SWIG_FLAGS "-builtin")
      ENDIF(PYTHONLIBS_VERSION_STRING MATCHES "3.5")

	  SWIG_ADD_LIBRARY(multiresolutionimageinterface LANGUAGE python SOURCES multiresolutionimageinterface.i)
	  SWIG_LINK_LIBRARIES(multiresolutionimageinterface multiresolutionimageinterface annotation core) 
	  SWIG_LINK_LIBRARIES(multiresolutionimageinterface ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES})
          IF(APPLE)
              target_include_directories(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PRIVATE ${DIAGPathology_BINARY_DIR}/annotation ${PYTHON_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR}/../../pkgs/numpy-1.13.1-py36_0/lib/python3.6/site-packages/numpy/core/include)
          ELSEIF(APPLE)
              target_include_directories(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PRIVATE ${DIAGPathology_BINARY_DIR}/annotation ${PYTHON_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR}/../Lib/site-packages/numpy/core/include)
          ENDIF(APPLE)
	  
	  SET_TARGET_PROPERTIES(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PROPERTIES FOLDER pathology/diag/io)
	  SET_TARGET_PROPERTIES(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PROPERTIES DEBUG_POSTFIX _d)
	  
	  # Cross-platform copying and installing instructions
	  INSTALL(FILES ${PROJECT_BINARY_DIR}/multiresolutionimageinterface.py DESTINATION bin)

      INSTALL(TARGETS ${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME}
	  	EXPORT asap-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
      )
	ENDIF(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON AND SWIG_FOUND)  
ENDIF(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)

INSTALL(FILES ${MULTIRESOLUTIONIMAGEINTERFACE_HS} DESTINATION include/multiresolutionimageinterface)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/multiresolutionimageinterface_export.h DESTINATION include/multiresolutionimageinterface)

INSTALL(FILES ${EXTERNAL_OPENSLIDE_HS} DESTINATION include/multiresolutionimageinterface)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/openslidefileformat_export.h DESTINATION include/multiresolutionimageinterface)

INSTALL(TARGETS multiresolutionimageinterface
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(TARGETS openslidefileformat
  EXPORT asap-targets
  RUNTIME DESTINATION bin/formats
  LIBRARY DESTINATION bin/formats
  ARCHIVE DESTINATION lib
)