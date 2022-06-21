set(WSIThreshold_src
    WSIThreshold.cpp
)

add_executable(WSIThreshold ${WSIThreshold_src})
set_target_properties(WSIThreshold PROPERTIES DEBUG_POSTFIX _d)
target_link_libraries(WSIThreshold wholeslidefilters multiresolutionimageinterface)

IF(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")

  install(TARGETS WSIThreshold 
          RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
          LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
          ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )

ELSE(APPLE)
  install(TARGETS WSIThreshold 
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
  )
ENDIF(APPLE)

if(WIN32)
  set_target_properties(WSIThreshold  PROPERTIES FOLDER executables)   
endif(WIN32)
