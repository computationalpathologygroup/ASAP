add_executable(MultiResImageConverter MultiResImageConverter.cpp)
set_target_properties(MultiResImageConverter PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(MultiResImageConverter PRIVATE multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(MultiResImageConverter PRIVATE -DBOOST_ALL_DYN_LINK)
install(
  TARGETS MultiResImageConverter
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

if(WIN32)
  set_target_properties(MultiResImageConverter PROPERTIES FOLDER executables)   
endif(WIN32)
