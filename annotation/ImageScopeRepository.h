#ifndef ANNOTATIONIMAGESCOPESERVICE_H
#define ANNOTATIONIMAGESCOPESERVICE_H

#include "annotation_export.h"
#include "Repository.h"

class ANNOTATION_EXPORT ImageScopeRepository : public Repository {
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