add_executable(MultiResImageConverter MultiResImageConverter.cpp)
set_target_properties(MultiResImageConverter PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(MultiResImageConverter PRIVATE multiresolutionimageinterface Boost::disable_autolinking Boost::program_options)
target_compile_definitions(MultiResImageConverter PRIVATE -DBOOST_ALL_DYN_LINK)
IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")
 
  install(
    TARGETS MultiResImageConverter
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
    LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
    ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )

ELSE(APPLE)

  install(
    TARGETS MultiResImageConverter
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
ENDIF(APPLE)
if(WIN32)
  set_target_properties(MultiResImageConverter PROPERTIES FOLDER executables)   
endif(WIN32)
