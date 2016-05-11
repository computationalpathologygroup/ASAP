#ifndef ANNOTATIONSERVICE_H
#define ANNOTATIONSERVICE_H
#include <vector>
#include <string>
#include <memory>
#include "annotation_export.h"

class AnnotationList;
class Repository;

class ANNOTATION_EXPORT AnnotationService {
public:
	AnnotationService();
	~AnnotationService();

  std::shared_ptr<AnnotationList> getList() const;

  std::shared_ptr<Repository> getRepository() const;
  bool loadRepositoryFromFile(const std::string& source);
  bool saveRepositoryToFile(const std::string& source);

private:
  std::shared_ptr<AnnotationList> _list;
  std::shared_ptr<Repository> _repo;

  bool load();
  bool save();

};

#endif