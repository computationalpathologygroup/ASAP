set(CORE_SRC filetools.cpp PathologyEnums.cpp ImageSource.cpp Patch.hpp Box.cpp Point.cpp ProgressMonitor.cpp CmdLineProgressMonitor.cpp stringconversion.cpp)
set(CORE_HEADERS filetools.h PathologyEnums.h ImageSource.h Patch.h Patch.hpp Box.h Point.h ProgressDisplay.hpp ProgressMonitor.h CmdLineProgressMonitor.h stringconversion.h)

add_library(core SHARED ${CORE_SRC} ${CORE_HEADERS})
generate_export_header(core)
target_include_directories(core PUBLIC $<BUILD_INTERFACE:${DIAGPathology_SOURCE_DIR}> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${DIAGPathology_BINARY_DIR}> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/core> PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(core PRIVATE cxx_generalized_initializers)
set_target_properties(core PROPERTIES DEBUG_POSTFIX _d)

if(APPLE)
  set(prefix "ASAP.app/Contents")
  set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
  set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
  set(INSTALL_CMAKE_DIR "${prefix}/Resources")

  install(TARGETS core
    EXPORT asap-targets
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
    LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
    ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
  )

ELSE(APPLE)
  install(TARGETS core
    EXPORT asap-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
ENDIF(APPLE)

install(FILES ${CORE_HEADERS} DESTINATION include/core)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/core_export.h DESTINATION include/core)