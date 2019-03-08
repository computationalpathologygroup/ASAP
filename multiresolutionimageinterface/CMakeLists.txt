option(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT "Builds support for Olympus VSI files; requires DCMTK ijg8.lib" OFF)

set(MULTIRESOLUTIONIMAGEINTERFACE_HS
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

set(MULTIRESOLUTIONIMAGEINTERFACE_SRCS 
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

set(OPENSLIDE_SUPPORT_HS
    OpenSlideImage.h
	OpenSlideImageFactory.h
)

set(VSI_SUPPORT_HS
    VSIImage.h
	VSIImageFactory.h
)

set(OPENSLIDE_SUPPORT_SRCS
    OpenSlideImage.cpp
	OpenSlideImageFactory.cpp
)

set(VSI_SUPPORT_SRCS
    VSIImage.cpp
	VSIImageFactory.cpp
)

if(WIN32)
add_definitions(-DNOMINMAX)
endif(WIN32)

add_library(jpeg2kcodec SHARED JPEG2000Codec.h JPEG2000Codec.cpp)
generate_export_header(jpeg2kcodec)
set_target_properties(jpeg2kcodec PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(jpeg2kcodec PRIVATE openjp2)
target_include_directories(jpeg2kcodec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/multiresolutionimageinterface> PRIVATE ${DIAGPathology_SOURCE_DIR} ${DIAGPathology_BINARY_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${OPENJPEG_INCLUDE_DIRS})

# Get info from PugiXML for the headers
get_target_property(PUGIXML_INCLUDE_DIR pugixml INTERFACE_INCLUDE_DIRECTORIES)

if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
  # Required for lossless JPEG compression used in VSIs
  find_package(DCMTKJPEG REQUIRED)
  list(APPEND MULTIRESOLUTIONIMAGEINTERFACE_SRCS ${VSI_SUPPORT_SRCS})
  list(APPEND MULTIRESOLUTIONIMAGEINTERFACE_HS ${VSI_SUPPORT_HS})
endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)

add_library(multiresolutionimageinterface SHARED ${MULTIRESOLUTIONIMAGEINTERFACE_SRCS} ${MULTIRESOLUTIONIMAGEINTERFACE_HS} ${VSI_SOURCE_HS} ${VSI_SOURCE_SRCS})
target_include_directories(multiresolutionimageinterface PUBLIC $<BUILD_INTERFACE:${DIAGPathology_SOURCE_DIR}> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/multiresolutionimageinterface> PRIVATE ${PUGIXML_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})
target_link_libraries(multiresolutionimageinterface PUBLIC core PRIVATE jpeg2kcodec libtiff Boost::disable_autolinking Boost::thread)
target_compile_definitions(multiresolutionimageinterface PRIVATE PUGIXML_HEADER_ONLY)
generate_export_header(multiresolutionimageinterface)
set_target_properties(multiresolutionimageinterface PROPERTIES DEBUG_POSTFIX _d)

if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
  # Required for lossless JPEG compression used in VSIs
  find_package(DCMTKJPEG REQUIRED)
  target_include_directories(multiresolutionimageinterface PRIVATE ${DCMTKJPEG_INCLUDE_DIR})
  target_link_libraries(multiresolutionimageinterface PRIVATE ${DCMTKJPEG_LIBRARY})
  target_compile_definitions(multiresolutionimageinterface PRIVATE HAS_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)

# Add the OpenSlide-plugin
add_library(openslidefileformat SHARED ${OPENSLIDE_SUPPORT_SRCS} ${OPENSLIDE_SUPPORT_HS})
generate_export_header(openslidefileformat)
target_include_directories(openslidefileformat PRIVATE ${OPENSLIDE_INCLUDE_DIR})
target_link_libraries(openslidefileformat PUBLIC multiresolutionimageinterface PRIVATE ${OPENSLIDE_LIBRARY} core Boost::disable_autolinking Boost::thread)
set_target_properties(openslidefileformat PROPERTIES DEBUG_POSTFIX _d)

if(WIN32)  
  set_target_properties(multiresolutionimageinterface PROPERTIES FOLDER multiresolutionimageinterface)    
  set_target_properties(jpeg2kcodec PROPERTIES FOLDER multiresolutionimageinterface)    
  set_target_properties(openslidefileformat PROPERTIES FOLDER multiresolutionimageinterface)
endif(WIN32)

option(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON "Wrap multi-resolution image reading in Python" OFF)
if(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)
  add_definitions(-DSWIG_PYTHON_INTERPRETER_NO_DEBUG)
  find_package(PythonLibs REQUIRED)
  find_package(SWIG REQUIRED)
  find_package(NumPy REQUIRED)
  include (${SWIG_USE_FILE})
  if(SWIG_FOUND)
    if(NOT WIN32)
      set(CMAKE_SWIG_FLAGS "-I${Boost_INCLUDE_DIR}")
    endif(NOT WIN32)     
    mark_as_advanced(SWIG_DIR SWIG_VERSION)
  endif(SWIG_FOUND)

	# Wrapping part for Python
	if (WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON AND SWIG_FOUND)   
	  # Set module extension, platform specific
	  if(WIN32)
		set(module_extension ".pyd")
	  elseif(APPLE)
	    set(module_extension ".dylib")
	  else(WIN32)
		set(module_extension ".so")  
	  endif(WIN32)
	  
	  set_source_files_properties(multiresolutionimageinterface.i PROPERTIES CPLUSPLUS ON)
	  set_source_files_properties(multiresolutionimageinterface.i PROPERTIES SWIG_FLAGS "-builtin")
      if(PYTHONLIBS_VERSION_STRING MATCHES "3.?")
	      set_property(SOURCE multiresolutionimageinterface.i PROPERTY SWIG_FLAGS "-py3")
      endif(PYTHONLIBS_VERSION_STRING MATCHES "3.?")

	  swig_add_library(multiresolutionimageinterface LANGUAGE python SOURCES multiresolutionimageinterface.i)
	  swig_link_libraries(multiresolutionimageinterface multiresolutionimageinterface annotation core) 
	  swig_link_libraries(multiresolutionimageinterface ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES})
      target_include_directories(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${DIAGPathology_BINARY_DIR}/annotation ${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR})
	  set_target_properties(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PROPERTIES FOLDER multiresolutionimageinterface)
	  set_target_properties(${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME} PROPERTIES DEBUG_POSTFIX _d)
	  
	  # Cross-platform copying and installing instructions
	  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/multiresolutionimageinterface.py DESTINATION bin)

      install(TARGETS ${SWIG_MODULE_multiresolutionimageinterface_REAL_NAME}
	  	EXPORT asap-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
      )
	endif(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON AND SWIG_FOUND)  
endif(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)

install(FILES ${MULTIRESOLUTIONIMAGEINTERFACE_HS} DESTINATION include/multiresolutionimageinterface)
install(FILES JPEG2000Codec.h DESTINATION include/multiresolutionimageinterface)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/multiresolutionimageinterface_export.h DESTINATION include/multiresolutionimageinterface)

install(FILES ${EXTERNAL_OPENSLIDE_HS} DESTINATION include/multiresolutionimageinterface)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openslidefileformat_export.h DESTINATION include/multiresolutionimageinterface)

install(TARGETS multiresolutionimageinterface
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(TARGETS jpeg2kcodec
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(TARGETS openslidefileformat
  EXPORT asap-targets
  RUNTIME DESTINATION bin/formats
  LIBRARY DESTINATION bin/formats
  ARCHIVE DESTINATION lib
)