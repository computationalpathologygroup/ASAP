# - Find Kakadu library
# Find the native Kakadu includes and library
# This module defines
#  KAKADU_INCLUDE_DIR, where to find OPENSLIDE.h, etc.
#  KAKADU_LIBRARY, libraries to link against to use Kakadu.
#  KAKADU_LIBRARY_DEBUG, debug libraries to link against to use Kakadu.
#  KAKADU_FOUND, If false, do not try to use KAKADU.

find_path(KAKADU_INCLUDE_DIR kdu_region_decompressor.h)
find_library(KAKADU_LIBRARY NAMES kdu_v7AR)
find_library(KAKADU_LIBRARY_DEBUG NAMES kdu_v7AD)
find_library(KAKADU_AUX_LIBRARY NAMES kdu_a7AR)
find_library(KAKADU_AUX_LIBRARY_DEBUG NAMES kdu_a7AD)
get_filename_component(KAKADU_LIBRARY_DIR ${KAKADU_LIBRARY} DIRECTORY)

# handle the QUIETLY and REQUIRED arguments and set TIFF_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(KAKADU
                                  REQUIRED_VARS KAKADU_LIBRARY KAKADU_AUX_LIBRARY KAKADU_INCLUDE_DIR)
                                  
if(KAKADU_FOUND)
  set(KAKADU_LIBRARIES ${KAKADU_LIBRARY} PARENT_SCOPE)
  set(KAKADU_LIBRARIES_DEBUG ${KAKADU_LIBRARY_DEBUG} PARENT_SCOPE)
  set(KAKADU_RUNTIME_DIR ${KAKADU_LIBRARY_DIR}/../bin PARENT_SCOPE)
endif()

mark_as_advanced(KAKADU_INCLUDE_DIR KAKADU_LIBRARY KAKADU_LIBRARY_DEBUG)
