PROJECT(NUCLEISEGMENTATION)
SET(NucleiSegmentation_SRCS FRST.h FRST.cpp)

ADD_LIBRARY(NucleiSegmentation SHARED ${NucleiSegmentation_SRCS})
TARGET_INCLUDE_DIRECTORIES(
  NucleiSegmentation
  PRIVATE
  ${DIAGPathology_SOURCE_DIR}  
  ${OpenCV_INCLUDE_DIRS}
  ${DIAGPathology_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
TARGET_LINK_LIBRARIES(NucleiSegmentation PRIVATE ${OpenCV_LIBS})
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
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/nucleisegmentation_export.h DESTINATION include/imgproc/NucleiSegmentation)