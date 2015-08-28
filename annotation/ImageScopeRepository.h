#ifndef ANNOTATIONIMAGESCOPESERVICE_H
#define ANNOTATIONIMAGESCOPESERVICE_H

#include "config/pathology_config.h"
#include "Repository.h"

class EXPORT_PATHOLOGYANNOTATION ImageScopeRepository : public Repository {
public:
  ImageScopeRepository(AnnotationList* list);

  virtual bool save() const;
  virtual bool load();
};

#endif