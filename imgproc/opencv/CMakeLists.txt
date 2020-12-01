set(OPENCVFILTERS_SRCS 
    DIAGPathologyOpenCVBridge.h
    NucleiDetectionFilter.h
    NucleiDetectionFilter.cpp
)

add_library(opencvfilters SHARED ${OPENCVFILTERS_SRCS})
target_include_directories(opencvfilters PUBLIC ${OpenCV_INCLUDE_DIRS} $<BUILD_INTERFACE:${DIAGPathology_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)
target_link_libraries(opencvfilters PUBLIC core ${OpenCV_LIBS})

if(WIN32)  
  set_target_properties(opencvfilters PROPERTIES FOLDER imgproc)
endif(WIN32)

install(FILES DIAGPathologyOpenCVBridge.h NucleiDetectionFilter.h DESTINATION include/imgproc/opencvfilters)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")

  install(TARGETS opencvfilters
          RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
          LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
          ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )
ELSE(APPLE)
  install(TARGETS opencvfilters
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
  )

ENDIF(APPLE)