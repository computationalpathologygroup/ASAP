set(WSIThreshold_src
    WSIThreshold.cpp
)

add_executable(WSIThreshold ${WSIThreshold_src})
set_target_properties(WSIThreshold PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(WSIThreshold wholeslidefilters multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(WSIThreshold PRIVATE -DBOOST_ALL_DYN_LINK)

install(TARGETS WSIThreshold 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(WSIThreshold  PROPERTIES FOLDER executables)   
endif(WIN32)
