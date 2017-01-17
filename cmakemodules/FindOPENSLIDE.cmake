# - Find OPENSLIDE library
# Find the native OPENSLIDE includes and library
# This module defines
#  OPENSLIDE_INCLUDE_DIR, where to find OPENSLIDE.h, etc.
#  OPENSLIDE_LIBRARY, libraries to link against to use OPENSLIDE.
#  OPENSLIDE_FOUND, If false, do not try to use OPENSLIDE.

find_path(OPENSLIDE_INCLUDE_DIR openslide.h)
find_library(OPENSLIDE_LIBRARY NAMES openslide)

# handle the QUIETLY and REQUIRED arguments and set TIFF_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENSLIDE
                                  REQUIRED_VARS OPENSLIDE_LIBRARY OPENSLIDE_INCLUDE_DIR)

if(OPENSLIDE_FOUND)
  set(OPENSLIDE_LIBRARIES ${OPENSLIDE_LIBRARY} )
endif()

mark_as_advanced(OPENSLIDE_INCLUDE_DIR OPENSLIDE_LIBRARY)
