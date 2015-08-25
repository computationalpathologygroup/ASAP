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
  bool loadRepositoryFromFile(const std::string& source);
  bool saveRepositoryToFile(const std::string& source);

private:
  AnnotationList* _list;
  Repository* _repo;

  bool load();
  bool save();

};

#endif