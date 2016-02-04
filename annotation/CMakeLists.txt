PROJECT(Annotation)
INCLUDE_DIRECTORIES(
  .
  ../
  ../io/multiresolutionimageinterface
  ${Boost_INCLUDE_DIRS}
  ${PugiXML_INCLUDE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
ADD_DEFINITIONS(${Boost_DEFINITIONS})
ADD_DEFINITIONS(-DBUILD_PATHOLOGYANNOTATION)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET(ANNOTATION_HEADERS
  Annotation.h
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
TARGET_LINK_LIBRARIES(annotation pugixml core multiresolutionimageinterface)
SET_TARGET_PROPERTIES(annotation PROPERTIES DEBUG_POSTFIX _d)

INSTALL(TARGETS annotation
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(FILES ${ANNOTATION_HEADERS} DESTINATION include/annotation)

IF(WIN32)
  SET_TARGET_PROPERTIES(annotation PROPERTIES FOLDER pathology/diag)   
ENDIF(WIN32)