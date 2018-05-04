PROJECT(ANNOTATION)
SET(ANNOTATION_HEADERS
  Annotation.h
  AnnotationBase.h
  AnnotationToMask.h
  AnnotationGroup.h
  AnnotationList.h
  AnnotationService.h
  XmlRepository.h
  NDPARepository.h
  ImageScopeRepository.h
  Repository.h
)

SET(ANNOTATION_SOURCE
  Annotation.cpp
  AnnotationBase.cpp
  AnnotationGroup.cpp
  AnnotationToMask.cpp
  AnnotationList.cpp
  AnnotationService.cpp
  XmlRepository.cpp
  NDPARepository.cpp
  ImageScopeRepository.cpp
  Repository.cpp
)

add_library(annotation SHARED ${ANNOTATION_SOURCE} ${ANNOTATION_HEADERS})
target_compile_definitions(annotation PRIVATE "PUGIXML_HEADER_ONLY")
generate_export_header(annotation)
get_target_property(PUGIXML_INCLUDE_DIR pugixml INTERFACE_INCLUDE_DIRECTORIES)
TARGET_INCLUDE_DIRECTORIES(
annotation
PRIVATE
  .
  ../
  ../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${PUGIXML_INCLUDE_DIR}
  ${DIAGPathology_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CORE_BINARY_DIR}
  ${MULTIRESOLUTIONIMAGEINTERFACE_BINARY_DIR}
)
TARGET_LINK_LIBRARIES(annotation PRIVATE core multiresolutionimageinterface)
SET_TARGET_PROPERTIES(annotation PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS annotation
  EXPORT asap-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES ${ANNOTATION_HEADERS} DESTINATION include/annotation)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/annotation_export.h DESTINATION include/annotation)

IF(WIN32)
  SET_TARGET_PROPERTIES(annotation PROPERTIES FOLDER pathology/diag)   
ENDIF(WIN32)