PROJECT(NUCLEISEGMENTATION)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
  ${DIAGPathology_SOURCE_DIR}  
  ${OpenCV_INCLUDE_DIRS}
  ${DIAGPathology_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

SET(NucleiSegmentation_SRCS FRST.h FRST.cpp)

ADD_LIBRARY(NucleiSegmentation SHARED ${NucleiSegmentation_SRCS})
IF(OpenCV_VERSION EQUAL "3.1.0")
TARGET_LINK_LIBRARIES(NucleiSegmentation optimized ${OpenCV_LIB_DIR_OPT}/opencv_world310.lib debug ${OpenCV_LIB_DIR_DBG}/opencv_world310d.lib)
ELSE()
TARGET_LINK_LIBRARIES(NucleiSegmentation ${OpenCV_LIBS})
ENDIF()
generate_export_header(NucleiSegmentation)

SET_TARGET_PROPERTIES(NucleiSegmentation PROPERTIES DEBUG_POSTFIX _d)

IF(WIN32)  
  SET_TARGET_PROPERTIES(NucleiSegmentation PROPERTIES FOLDER pathology/diag/imgproc)    
ENDIF(WIN32)

INSTALL(TARGETS NucleiSegmentation
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES FRST.h DESTINATION include/imgproc/NucleiSegmentation)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/NucleiSegmentation_export.h DESTINATION include/imgproc/NucleiSegmentation)