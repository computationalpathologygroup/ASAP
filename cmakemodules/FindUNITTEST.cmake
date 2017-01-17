# - Find UNITTEST library
# Find the UNITTEST includes and library
# This module defines
#  UNITTEST_INCLUDE_DIR, where to find unittest.h, etc.
#  UNITTEST_LIBRARIES, libraries to link against to use UNITTEST.
#  UNITTEST_FOUND, If false, do not try to use UNITTEST.
# also defined, but not for general use are
#  UNITTEST_LIBRARY, where to find the UNITTEST library.

find_path(UNITTEST_INCLUDE_DIR UnitTest++.h)
find_library(UNITTEST_LIBRARY NAMES UNITTEST)
find_library(UNITTEST_LIBRARY_DEBUG NAMES UNITTEST)

# handle the QUIETLY and REQUIRED arguments and set TIFF_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(UNITTEST
                                  REQUIRED_VARS UNITTEST_LIBRARY UNITTEST_LIBRARY_DEBUG UNITTEST_INCLUDE_DIR)

if(UNITTEST_FOUND)
  set( UNITTEST_LIBRARIES ${UNITTEST_LIBRARY} )
endif()

mark_as_advanced(UNITTEST_INCLUDE_DIR UNITTEST_LIBRARY)
