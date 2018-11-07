add_executable(WSIArithmetic WSIArithmetic.cpp)
set_target_properties(WSIArithmetic PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(WSIArithmetic PRIVATE wholeslidefilters multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(WSIArithmetic PRIVATE -DBOOST_ALL_DYN_LINK)

install(TARGETS WSIArithmetic 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(WSIArithmetic PROPERTIES FOLDER executables)   
endif(WIN32)
