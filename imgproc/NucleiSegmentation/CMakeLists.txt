PROJECT(NucleiSegmentation)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
  ${DIAGPathology_SOURCE_DIR}  
  ${OpenCV_INCLUDE_DIRS}
  ${DIAGPathology_BINARY_DIR}
)

IF(WIN32)
  ADD_DEFINITIONS(-DBUILD_NUCLEISEGMENTATION)
ENDIF(WIN32)

SET(NucleiSegmentation_SRCS FRST.h FRST.cpp)


ADD_LIBRARY(NucleiSegmentation SHARED ${NucleiSegmentation_SRCS})
TARGET_LINK_LIBRARIES(NucleiSegmentation ${OpenCV_LIBS})

SET_TARGET_PROPERTIES(NucleiSegmentation PROPERTIES DEBUG_POSTFIX _d)

IF(WIN32)  
  SET_TARGET_PROPERTIES(NucleiSegmentation PROPERTIES FOLDER pathology/diag/imgproc)    
ENDIF(WIN32)

INSTALL(TARGETS NucleiSegmentation
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES FRST.h DESTINATION include/NucleiSegmentation)