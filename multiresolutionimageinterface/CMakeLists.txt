option(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT "Builds support for Olympus VSI files; requires DCMTK ijg8.lib" ON)
option(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT "Builds support for DICOM files; requires DCMTK" ON)

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

set(DICOM_SUPPORT_HS
    DICOMImage.h
	DICOMImageFactory.h
    WSIDicomInstance.h
)

set(DICOM_SUPPORT_SRCS
    DICOMImage.cpp
	DICOMImageFactory.cpp
    WSIDicomInstance.cpp
)

if(WIN32)
add_definitions(-DNOMINMAX)
endif(WIN32)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")
ENDIF(APPLE)


add_library(jpeg2kcodec SHARED JPEG2000Codec.h JPEG2000Codec.cpp)
generate_export_header(jpeg2kcodec)
set_target_properties(jpeg2kcodec PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(jpeg2kcodec PRIVATE openjp2)
target_include_directories(jpeg2kcodec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/multiresolutionimageinterface> PRIVATE ${DIAGPathology_SOURCE_DIR} ${DIAGPathology_BINARY_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${OPENJPEG_INCLUDE_DIRS})

if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)
  # Required for lossless JPEG compression used in VSIs
  find_package(DCMTKJPEG REQUIRED)
  list(APPEND MULTIRESOLUTIONIMAGEINTERFACE_SRCS ${VSI_SUPPORT_SRCS})
  list(APPEND MULTIRESOLUTIONIMAGEINTERFACE_HS ${VSI_SUPPORT_HS})
endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT)

add_library(multiresolutionimageinterface SHARED ${MULTIRESOLUTIONIMAGEINTERFACE_SRCS} ${MULTIRESOLUTIONIMAGEINTERFACE_HS} ${VSI_SOURCE_HS} ${VSI_SOURCE_SRCS})
target_include_directories(multiresolutionimageinterface PUBLIC $<BUILD_INTERFACE:${DIAGPathology_SOURCE_DIR}> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/multiresolutionimageinterface> PRIVATE ${PugiXML_INCLUDE_DIR} ${TIFF_INCLUDE_DIR})
target_link_libraries(multiresolutionimageinterface PUBLIC core PRIVATE jpeg2kcodec libtiff)
IF(NOT WIN32)
  target_link_libraries(multiresolutionimageinterface PRIVATE dl)
ENDIF(NOT WIN32)
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
target_link_libraries(openslidefileformat PUBLIC multiresolutionimageinterface PRIVATE ${OPENSLIDE_LIBRARY} core)
set_target_properties(openslidefileformat PROPERTIES DEBUG_POSTFIX _d)

if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)
  find_package(DCMTK REQUIRED)
  add_library(dicomfileformat SHARED ${DICOM_SUPPORT_SRCS} ${DICOM_SUPPORT_HS})
  generate_export_header(dicomfileformat)
  target_include_directories(dicomfileformat PRIVATE ${DCMTK_INCLUDE_DIR})
  # Small workaround to support slightly older versions of DCMTK (such as in vcpkg)
  if(TARGET DCMTK::dcmdata)
    target_link_libraries(dicomfileformat PUBLIC multiresolutionimageinterface PRIVATE core DCMTK::dcmjpeg DCMTK::dcmimage DCMTK::dcmdata jpeg2kcodec)
  else(TARGET DCMTK::dcmdata)
    target_link_libraries(dicomfileformat PUBLIC multiresolutionimageinterface PRIVATE core dcmjpeg dcmimage dcmdata jpeg2kcodec)
  endif(TARGET DCMTK::dcmdata)
  set_target_properties(dicomfileformat PROPERTIES DEBUG_POSTFIX _d)
endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)

if(WIN32)  
  set_target_properties(multiresolutionimageinterface PROPERTIES FOLDER multiresolutionimageinterface)    
  set_target_properties(jpeg2kcodec PROPERTIES FOLDER multiresolutionimageinterface)    
  set_target_properties(openslidefileformat PROPERTIES FOLDER multiresolutionimageinterface)
  if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)
    set_target_properties(dicomfileformat PROPERTIES FOLDER multiresolutionimageinterface)
  endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)
endif(WIN32)

option(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON "Wrap multi-resolution image reading in Python" OFF)
if(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)
  add_definitions(-DSWIG_PYTHON_INTERPRETER_NO_DEBUG)

  # IF(APPLE)
    # set(Python_FIND_ABI "ANY" "ANY" "ANY")
    # find_package(Python3 COMPONENTS Interpreter Development)
  # ELSE(APPLE)
    find_package(PythonLibs REQUIRED)
  # ENDIF(APPLE)

  find_package(SWIG REQUIRED)
  find_package(NumPy REQUIRED)
  include (${SWIG_USE_FILE})
  if(SWIG_FOUND)
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

    swig_add_library(py_multiresolutionimageinterface LANGUAGE python SOURCES multiresolutionimageinterface.i)
    swig_link_libraries(py_multiresolutionimageinterface   multiresolutionimageinterface annotation core) 

	  IF(APPLE)
	    # Relevant CMake discussions:
	    # https://gitlab.kitware.com/cmake/cmake/-/issues/17664#note_643333
	    # https://gitlab.kitware.com/cmake/cmake/-/issues/18100
	    # Finally used:
	    # https://gitlab.kitware.com/cmake/cmake/-/issues/18536
	    # We need to weakly link to Python, otherwise the linker will add a hardlink to the python
	    # used during compiling. This will make the library not work for other people / other Python versions.
	    #
	    macro( target_link_libraries_with_dynamic_lookup target )
	      if ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
		      set_target_properties( ${target} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup" )
	      else()
		      target_link_libraries ( ${target} ${ARGN}  )
	      endif()
	    endmacro()

	    target_link_libraries_with_dynamic_lookup(py_multiresolutionimageinterface ${PYTHON_LIBRARIES})
	  ELSE(APPLE)
	    swig_link_libraries(py_multiresolutionimageinterface ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES})
	  ENDIF(APPLE)

    target_include_directories(py_multiresolutionimageinterface   PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${DIAGPathology_BINARY_DIR}/annotation ${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR})
	  set_target_properties(py_multiresolutionimageinterface  PROPERTIES FOLDER multiresolutionimageinterface)
	  set_target_properties(py_multiresolutionimageinterface  PROPERTIES OUTPUT_NAME multiresolutionimageinterface)
 	  set_target_properties(py_multiresolutionimageinterface  PROPERTIES ARCHIVE_OUTPUT_NAME _multiresolutionimageinterface)
	  set_target_properties(py_multiresolutionimageinterface  PROPERTIES DEBUG_POSTFIX _d)
      IF(WIN32)
      target_compile_options(py_multiresolutionimageinterface PRIVATE "/wd6011;")
      endif(WIN32)
    # Cross-platform copying and installing instructions
    IF(APPLE)
	      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/multiresolutionimageinterface.py DESTINATION ${INSTALL_RUNTIME_DIR})
        install(TARGETS py_multiresolutionimageinterface
        EXPORT asap-targets
          RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
          LIBRARY DESTINATION ${INSTALL_RUNTIME_DIR}
          ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
        )

    ELSE(APPLE)
	      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/multiresolutionimageinterface.py DESTINATION bin)

        install(TARGETS py_multiresolutionimageinterface
        EXPORT asap-targets
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION bin
          ARCHIVE DESTINATION lib
        )
    ENDIF(APPLE)
	endif(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON AND SWIG_FOUND)  
endif(WRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON)

install(FILES ${MULTIRESOLUTIONIMAGEINTERFACE_HS} DESTINATION include/multiresolutionimageinterface)
install(FILES JPEG2000Codec.h DESTINATION include/multiresolutionimageinterface)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/multiresolutionimageinterface_export.h DESTINATION include/multiresolutionimageinterface)

install(FILES ${EXTERNAL_OPENSLIDE_HS} DESTINATION include/multiresolutionimageinterface)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openslidefileformat_export.h DESTINATION include/multiresolutionimageinterface)

if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)
  install(FILES ${DICOM_OPENSLIDE_HS} DESTINATION include/multiresolutionimageinterface)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dicomfileformat_export.h DESTINATION include/multiresolutionimageinterface)
endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)

IF(APPLE)

  install(TARGETS multiresolutionimageinterface
    EXPORT asap-targets
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
    LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
    ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )

  install(TARGETS jpeg2kcodec
    EXPORT asap-targets
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
    LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
    ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )

  install(TARGETS openslidefileformat
    EXPORT asap-targets
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/formats
    LIBRARY DESTINATION ${INSTALL_RUNTIME_DIR}/formats
    ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )

  if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)
  install(TARGETS dicomfileformat
    EXPORT asap-targets
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/formats
    LIBRARY DESTINATION ${INSTALL_RUNTIME_DIR}/formats
    ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )
  endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)


ELSE(APPLE)

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
  if(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)
  install(TARGETS dicomfileformat
    EXPORT asap-targets
    RUNTIME DESTINATION bin/formats
    LIBRARY DESTINATION bin/formats
    ARCHIVE DESTINATION lib
  )
  endif(BUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT)


ENDIF(APPLE)
