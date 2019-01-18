set(WSIConnectedComponents_src
    WSIConnectedComponents.cpp
)

add_executable(WSIConnectedComponents ${WSIConnectedComponents_src})
set_target_properties(WSIConnectedComponents PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(WSIConnectedComponents PRIVATE wholeslidefilters multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(WSIConnectedComponents PRIVATE -DBOOST_ALL_DYN_LINK)

install(TARGETS WSIConnectedComponents
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(WSIConnectedComponents PROPERTIES FOLDER executables)   
endif(WIN32)
