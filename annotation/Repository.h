#ifndef REPOSITORY_H
#define REPOSITORY_H

#include <string>
#include <vector>
#include <memory>
#include "annotation_export.h"

class AnnotationList;

class ANNOTATION_EXPORT Repository
{
public:
  Repository(const std::shared_ptr<AnnotationList>& list);
  virtual ~Repository();

  void setSource(const std::string& sourcePath);
  bool load();
  virtual bool save() const = 0;

protected:

  virtual bool loadFromRepo() = 0;
  std::shared_ptr<AnnotationList> _list;
  std::string _source;
};

#endif