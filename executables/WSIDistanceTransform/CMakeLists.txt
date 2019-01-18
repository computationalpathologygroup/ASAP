set(WSIDistanceTransform_src
    WSIDistanceTransform.cpp
)

add_executable(WSIDistanceTransform ${WSIDistanceTransform_src})
set_target_properties(WSIDistanceTransform PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(WSIDistanceTransform PRIVATE wholeslidefilters multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(WSIDistanceTransform PRIVATE -DBOOST_ALL_DYN_LINK)

install(TARGETS WSIDistanceTransform 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(WSIDistanceTransform  PROPERTIES FOLDER executables)   
endif(WIN32)
