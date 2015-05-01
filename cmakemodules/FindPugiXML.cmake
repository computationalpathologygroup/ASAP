# - Find PugiXML library
# Find the native PugiXML includes and library
# This module defines
#  PugiXML_INCLUDE_DIRS, where to find PugiXML.h, etc.
#  PugiXML_LIBRARY, libraries to link against to use PugiXML.
#  PugiXML_FOUND, If false, do not try to use PugiXML.

find_path(PugiXML_INCLUDE_DIR pugixml.hpp)
find_library(PugiXML_LIBRARY NAMES pugixml)

# handle the QUIETLY and REQUIRED arguments and set PugiXML_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PugiXML
                                  REQUIRED_VARS PugiXML_LIBRARY PugiXML_INCLUDE_DIR)

if(PugiXML_FOUND)
  set(PugiXML_LIBRARIES ${PugiXML_LIBRARY} )
endif()

mark_as_advanced(PugiXML_INCLUDE_DIR PugiXML_LIBRARY)
