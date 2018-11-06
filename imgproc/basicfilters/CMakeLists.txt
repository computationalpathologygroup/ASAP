set(BASICFILTERS_SRCS 
    FilterBase.h
    FilterBase.cpp
    ImageFilter.h
    ImageFilter.cpp
    ColorDeconvolutionFilter.h
    ColorDeconvolutionFilter.cpp
)

add_library(basicfilters SHARED ${BASICFILTERS_SRCS})
target_include_directories(basicfilters PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/imgproc/basicfilters>)
target_link_libraries(basicfilters PUBLIC core)
generate_export_header(basicfilters)
set_target_properties(basicfilters PROPERTIES DEBUG_POSTFIX _d)

IF(WIN32)  
  set_target_properties(basicfilters PROPERTIES FOLDER imgproc)    
ENDIF(WIN32)


install(FILES FilterBase.h ColorDeconvolutionFilter.h ImageFilter.h DESTINATION include/imgproc/basicfilters)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/basicfilters_export.h DESTINATION include/imgproc/basicfilters)

install(TARGETS basicfilters
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)