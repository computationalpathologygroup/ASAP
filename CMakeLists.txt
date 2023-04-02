cmake_minimum_required(VERSION 3.15.0)
project(DIAGPathology)

set(CURRENT_MAJOR_VERSION 2)
set(CURRENT_MINOR_VERSION 2)
set(CURRENT_PATCH_VERSION 0)
set(ASAP_VERSION ${CURRENT_MAJOR_VERSION}.${CURRENT_MINOR_VERSION}.${CURRENT_PATCH_VERSION})

include (GenerateExportHeader)

configure_file(
  "${PROJECT_SOURCE_DIR}/config/ASAPMacros.h.in"
  "${PROJECT_BINARY_DIR}/config/ASAPMacros.h"
)

set(CMAKE_MODULE_PATH ${DIAGPathology_SOURCE_DIR}/cmakemodules)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT WIN32)
	set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
	set(CMAKE_MACOSX_RPATH TRUE)
ENDIF(NOT WIN32)

set(CMAKE_CXX_STANDARD 17)

# Set compile definitions to ignore some warnings
if(WIN32)
    set(DEFAULT_COMPILE_DEFINITIONS ${DEFAULT_COMPILE_DEFINITIONS}
        /D"_SCL_SECURE_NO_WARNINGS"  # Calling any one of the potentially unsafe methods in the Standard C++ Library
        /D"_CRT_SECURE_NO_WARNINGS"  # Calling any one of the potentially unsafe methods in the CRT Library
		/wd4251

    )
	add_definitions(${DEFAULT_COMPILE_DEFINITIONS})
endif()

# OpenJPEG for JPEG2000 codec
find_package(OpenJPEG REQUIRED)

# TIFF is the main multi-tiled format used in this package
find_package(TIFF REQUIRED)

# JPEG for the codecs
find_package(JPEG REQUIRED)

# ZLIB for codec support
find_package(ZLIB REQUIRED)

# PugiXML required for LIF file format and reading/writing annotations
find_package(PugiXML REQUIRED)

# Required for additional format support
find_package(OPENSLIDE REQUIRED)

# Build options for the pathology projects
option(BUILD_EXECUTABLES "Builds executables" OFF)
option(BUILD_ASAP "Builds the ASAP viewer" OFF)
option(BUILD_IMAGEPROCESSING "Builds image processing routines" OFF)
option(BUILD_TESTS "Builds tests" OFF)
option(PACKAGE_ON_INSTALL "Copies dependent DLLs and packages install" OFF)

if(BUILD_IMAGEPROCESSING)
  find_package(OpenCV REQUIRED)
endif()

add_subdirectory(core)
add_subdirectory(multiresolutionimageinterface)
add_subdirectory(annotation)

IF(BUILD_EXECUTABLES)
  add_subdirectory(executables)
ENDIF()

IF(BUILD_IMAGEPROCESSING)
  add_subdirectory(imgproc)
ENDIF()

IF(BUILD_ASAP)
  find_package(Qt6 REQUIRED COMPONENTS Core Widgets Concurrent UiTools)
  qt_standard_project_setup()
  option(BUILD_PYSIDE6_WRAPPING "Builds PyASAP wrapping" OFF)
  add_subdirectory(ASAP)
  if(BUILD_PYSIDE6_WRAPPING)
    add_subdirectory(pysidewrapping)
  endif(BUILD_PYSIDE6_WRAPPING)
ENDIF()

add_subdirectory(buildtools)

if(APPLE)
  set(prefix "${CMAKE_INSTALL_PREFIX}/ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR ${prefix}/MacOS)
  set(INSTALL_CMAKE_DIR ${prefix}/Resources)
  set(INSTALL_INCLUDE_DIR ${prefix}/Frameworks)
  set(INSTALL_BINARY_DIR ${INSTALL_RUNTIME_DIR})
  set(INSTALL_LIBRARY_DIR ${prefix}/Frameworks)
ELSE(APPLE)
  set(INSTALL_CMAKE_DIR ${CMAKE_INSTALL_PREFIX}/cmake)
  set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
  set(INSTALL_BINARY_DIR ${CMAKE_INSTALL_PREFIX}/bin)
  set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib)
ENDIF(APPLE)

set(INSTALL_CMAKE_DIR ${CMAKE_INSTALL_PREFIX}/cmake)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(INSTALL_BINARY_DIR ${CMAKE_INSTALL_PREFIX}/bin)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib)

configure_file(ASAPConfig.cmake.in
               "${PROJECT_BINARY_DIR}/ASAPConfig.cmake" @ONLY)
configure_file(ASAPConfigVersion.cmake.in
               "${PROJECT_BINARY_DIR}/ASAPConfigVersion.cmake" @ONLY)
 
# Install the files for version configuration
install(FILES
        "${PROJECT_BINARY_DIR}/ASAPConfig.cmake"
        "${PROJECT_BINARY_DIR}/ASAPConfigVersion.cmake"
        DESTINATION cmake)
 
# Install the export set for use with the install-tree
install(EXPORT asap-targets DESTINATION cmake)

if(PACKAGE_ON_INSTALL)
  set(CPACK_PACKAGE_NAME "ASAP")
  set(CPACK_PACKAGE_VENDOR "Diagnostic Image Analysis Group")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ASAP - Automated Slide Analysis Platform")
  set(CPACK_PACKAGE_DESCRIPTION "This package provides viewing and analysis tools for whole slide images.")
  set(CPACK_PACKAGE_VERSION "${CURRENT_MAJOR_VERSION}.${CURRENT_MINOR_VERSION}")
  set(CPACK_PACKAGE_VERSION_MAJOR ${CURRENT_MAJOR_VERSION})
  set(CPACK_PACKAGE_VERSION_MINOR ${CURRENT_MINOR_VERSION})
  set(CPACK_PACKAGE_VERSION_PATCH ${CURRENT_PATCH_VERSION})
if(WIN32)
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  set(CPACK_NSIS_DISPLAY_NAME "ASAP ${CPACK_PACKAGE_VERSION}")
  if(BUILD_WORKLIST_INTERFACE)
    set(CPACK_PACKAGE_EXECUTABLES ASAP "ASAP" ASAP_browser "ASAP Browser")
  else(BUILD_WORKLIST_INTERFACE)
    set(CPACK_PACKAGE_EXECUTABLES asap "ASAP")
  endif(BUILD_WORKLIST_INTERFACE)
  set(CPACK_NSIS_MUI_ICON ${DIAGPathology_SOURCE_DIR}/ASAP/application.ico)
  set(CPACK_NSIS_MUI_UNIICON ${DIAGPathology_SOURCE_DIR}/ASAP/application.ico)
elseif(APPLE)
  set(CPACK_GENERATOR "DragNDrop")
  set(CPACK_BUNDLE_NAME ASAP)
  SET(MACOSX_BUNDLE_ICON_FILE ${DIAGPathology_SOURCE_DIR}/ASAP/application.icns)
  SET(MACOSX_GUI_ICON ${DIAGPathology_SOURCE_DIR}/ASAP/application.icns)
  set(CPACK_PACKAGE_ICON ${DIAGPathology_SOURCE_DIR}/ASAP/application.icns)
  set(CPACK_DMG_DISABLE_APPLICATIONS_SYMLINK ON)
  if(BUILD_WORKLIST_INTERFACE)
    set(CPACK_PACKAGE_EXECUTABLES ASAP "ASAP" ASAP_browser "ASAP Browser")
  else(BUILD_WORKLIST_INTERFACE)
    set(CPACK_PACKAGE_EXECUTABLES asap "ASAP")
  endif(BUILD_WORKLIST_INTERFACE)
  set(CPACK_PACKAGING_INSTALL_PREFIX /Applications/ASAP)
elseif(UNIX)
  set(CPACK_GENERATOR "DEB")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Geert Litjens") 
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "ASAP - Automated Slide Analysis Platform")
  set(CPACK_PACKAGING_INSTALL_PREFIX /opt/ASAP)
endif(WIN32)
include(CPack)
endif(PACKAGE_ON_INSTALL)
