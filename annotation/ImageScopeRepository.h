#ifndef ANNOTATIONIMAGESCOPESERVICE_H
#define ANNOTATIONIMAGESCOPESERVICE_H

#include "config/pathology_config.h"
#include "Repository.h"

class EXPORT_PATHOLOGYANNOTATION ImageScopeRepository : public Repository {
private:
  float _closingDistance;
  bool loadFromRepo();

public:
  ImageScopeRepository(const std::shared_ptr<AnnotationList>& list);

  virtual bool save() const;

  void setClosingDistance(const float& closingDistance);
  float getClosingDistance();
};

#endif