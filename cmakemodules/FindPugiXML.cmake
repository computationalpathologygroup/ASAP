# - Find PugiXML library
# Find the native PugiXML includes
# This module defines
#  PugiXML_INCLUDE_DIR, where to find PugiXML.hpp
#  PugiXML_FOUND, If false, do not try to use PugiXML.

find_path(PugiXML_INCLUDE_DIR pugixml.hpp)

# handle the QUIETLY and REQUIRED arguments and set PugiXML_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PugiXML
                                  REQUIRED_VARS PugiXML_INCLUDE_DIR)

mark_as_advanced(PugiXML_INCLUDE_DIR)
