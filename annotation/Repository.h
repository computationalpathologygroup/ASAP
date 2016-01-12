#ifndef REPOSITORY_H
#define REPOSITORY_H

#include <string>
#include <vector>
#include <memory>
#include "config/pathology_config.h"

class AnnotationList;

class EXPORT_PATHOLOGYANNOTATION Repository
{
public:
  Repository(const std::shared_ptr<AnnotationList>& list);
	virtual ~Repository();

  void setSource(const std::string& sourcePath);
	virtual bool load() = 0;
  virtual bool save() const = 0;

protected:
  std::shared_ptr<AnnotationList> _list;
  std::string _source;
};

#endif