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
  XmlRepository(const std::shared_ptr<AnnotationList>& list);
  virtual bool save() const;  

private :
  bool loadFromRepo();
  void saveAnnotation(const std::shared_ptr<Annotation>&, pugi::xml_node* node) const;
  void saveGroup(const std::shared_ptr<AnnotationGroup>& group, pugi::xml_node* node)  const;
};

#endif