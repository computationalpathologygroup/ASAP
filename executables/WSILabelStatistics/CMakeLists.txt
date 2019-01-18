set(WSILabelStatistics_src
    WSILabelStatistics.cpp
)

add_executable(WSILabelStatistics ${WSILabelStatistics_src})
set_target_properties(WSILabelStatistics PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(WSILabelStatistics wholeslidefilters multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(WSILabelStatistics PRIVATE -DBOOST_ALL_DYN_LINK)

install(TARGETS WSILabelStatistics 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(WSILabelStatistics  PROPERTIES FOLDER executables)   
endif(WIN32)
