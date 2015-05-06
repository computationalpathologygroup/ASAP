#ifndef ANNOTATIONXMLSERVICE_H
#define ANNOTATIONXMLSERVICE_H

#include "config/pathology_config.h"
#include "Repository.h"

class Annotation;
class AnnotationGroup;

namespace pugi {
  class xml_node;
}

class EXPORT_PATHOLOGYANNOTATION XmlRepository : public Repository {
public:
  XmlRepository(AnnotationList* list);
  virtual bool save() const;
  virtual bool load();

private :
  void saveAnnotation(const Annotation* annotation, pugi::xml_node* node) const;
  void saveGroup(const AnnotationGroup* group, pugi::xml_node* node)  const;
};

#endif