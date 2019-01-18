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
install(TARGETS opencvfilters
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)