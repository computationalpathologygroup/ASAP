set(ANNOTATION_HEADERS
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

set(ANNOTATION_SOURCE
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
target_include_directories(annotation PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/annotation> PRIVATE ${DIAGPathology_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${PUGIXML_INCLUDE_DIR})
target_link_libraries(annotation PUBLIC core PRIVATE  multiresolutionimageinterface)
set_target_properties(annotation PROPERTIES DEBUG_POSTFIX _d)

install(TARGETS annotation
        EXPORT asap-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(FILES ${ANNOTATION_HEADERS} DESTINATION include/annotation)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/annotation_export.h DESTINATION include/annotation)
