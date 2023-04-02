# - Find OpenJPEG library
# Find the native OpenJPEG includes and library
# This module defines
#  OpenJPEG_INCLUDE_DIR, where to find OpenJPEG.h, etc.
#  OpenJPEG_LIBRARY, libraries to link against to use OpenJPEG.
#  OpenJPEG_FOUND, If false, do not try to use OpenJPEG.

find_path(OpenJPEG_INCLUDE_DIR opj_config.h)
find_library(OpenJPEG_LIBRARY NAMES openjp2)

# handle the QUIETLY and REQUIRED arguments and set TIFF_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenJPEG
                                  REQUIRED_VARS OpenJPEG_LIBRARY OpenJPEG_INCLUDE_DIR)

if(OpenJPEG_FOUND)
  set(OpenJPEG_LIBRARIES ${OpenJPEG_LIBRARY} )

  if(NOT TARGET OpenJPEG::OpenJPEG)
    add_library(OpenJPEG::OpenJPEG UNKNOWN IMPORTED)
    if(OpenJPEG_INCLUDE_DIR)
      set_target_properties(OpenJPEG::OpenJPEG PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpenJPEG_INCLUDE_DIR}")
    endif()
    if(EXISTS "${OpenJPEG_LIBRARY}")
      set_target_properties(OpenJPEG::OpenJPEG PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${OpenJPEG_LIBRARY}")
    endif()
  endif()
endif()

mark_as_advanced(OpenJPEG_INCLUDE_DIR OpenJPEG_LIBRARY)
