#ifndef ANNOTATIONSERVICE_H
#define ANNOTATIONSERVICE_H
#include <vector>
#include <string>
#include "config/pathology_config.h"

class AnnotationList;
class Repository;

class EXPORT_PATHOLOGYANNOTATION AnnotationService {
public:
	AnnotationService();
	~AnnotationService();

  AnnotationList* getList() const;

  Repository* getRepository() const;
  void setRepositoryFromSourceFile(const std::string& source);

  bool load();
  bool save();

private:
  AnnotationList* _list;
  Repository* _repo;
};

#endif