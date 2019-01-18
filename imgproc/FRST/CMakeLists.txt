set(FRST_SRCS FRST.h FRST.cpp)

add_library(FRST SHARED ${FRST_SRCS})
target_include_directories(FRST PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/imgproc/FRST> PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FRST PRIVATE ${OpenCV_LIBS})
generate_export_header(FRST)
set_target_properties(FRST PROPERTIES DEBUG_POSTFIX _d)

if(WIN32)  
  SET_TARGET_PROPERTIES(FRST PROPERTIES FOLDER imgproc)    
endif(WIN32)

install(TARGETS FRST
        EXPORT asap-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(FILES FRST.h DESTINATION include/imgproc/FRST)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/frst_export.h DESTINATION include/imgproc/FRST)
