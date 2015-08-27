#ifndef REPOSITORY_H
#define REPOSITORY_H

#include <string>
#include <vector>
#include "config/pathology_config.h"

class AnnotationList;

class EXPORT_PATHOLOGYANNOTATION Repository
{
public:
  Repository(AnnotationList* list);
	virtual ~Repository();

  void setSource(const std::string& sourcePath);
	virtual bool load() = 0;
  virtual bool save() const = 0;

protected:
  AnnotationList* _list;
  std::string _source;
};

#endif