PROJECT(CORE)

SET(core_src filetools.cpp stringconversion.cpp PathologyEnums.cpp ImageSource.cpp Patch.hpp Box.cpp Point.cpp ProgressMonitor.cpp CmdLineProgressMonitor.cpp)
SET(core_headers filetools.h stringconversion.h PathologyEnums.h ImageSource.h Patch.h Patch.hpp Box.h Point.h ProgressMonitor.h CmdLineProgressMonitor.h)

ADD_LIBRARY(core SHARED ${core_src} ${core_headers})
generate_export_header(core)
target_link_libraries(core PRIVATE Boost::disable_autolinking Boost::filesystem Boost::system Boost::regex)

target_include_directories(core PRIVATE ${CORE_SOURCE_DIR} ${DIAGPathology_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${DIAGPathology_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

target_compile_features(core PRIVATE cxx_generalized_initializers)

SET_TARGET_PROPERTIES(core PROPERTIES DEBUG_POSTFIX _d)
IF(WIN32)  
  SET_TARGET_PROPERTIES(core PROPERTIES FOLDER diag)    
ENDIF(WIN32)

INSTALL(TARGETS core
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES ${core_headers} DESTINATION include/core)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/core_export.h DESTINATION include/core)

IF(WIN32)
  SET_TARGET_PROPERTIES(core PROPERTIES FOLDER pathology/diag)   
ENDIF(WIN32)
