#include "AnnotationService.h"
#include "AnnotationList.h"
#include "XmlRepository.h"
#include "NDPARepository.h"
#include "ImageScopeRepository.h"

AnnotationService::AnnotationService() :
  _list(NULL),
  _repo(NULL)
{
  _list = std::make_shared<AnnotationList>();
}

AnnotationService::~AnnotationService() {
}

std::shared_ptr<AnnotationList> AnnotationService::getList() const {
  return _list;
}

std::shared_ptr<Repository> AnnotationService::getRepository() const {
  return _repo;
}

bool AnnotationService::loadRepositoryFromFile(const std::string& source) {
  if (source.rfind(std::string(".xml")) != source.npos) {
    _repo = std::make_shared<XmlRepository>(_list);
    _repo->setSource(source);
    if (!_repo->load()) {
      _list->removeAllAnnotations();
      _list->removeAllGroups();
      _repo = std::make_shared<ImageScopeRepository>(_list);
      _repo->setSource(source);
    }
  }
  else if (source.rfind(std::string(".ndpa")) != source.npos) {
    _repo = std::make_shared<NDPARepository>(_list);
    _repo->setSource(source);
  }
  return _repo->load();
}

bool AnnotationService::saveRepositoryToFile(const std::string& source) {
  if (source.rfind(std::string(".xml")) != source.npos) {
    _repo = std::make_shared<XmlRepository>(_list);
    _repo->setSource(source);
  }
  else if (source.rfind(std::string(".ndpa")) != source.npos) {
    _repo = std::make_shared<NDPARepository>(_list);
    _repo->setSource(source);
  }
  return _repo->save();
}

bool AnnotationService::load() {
  if (_repo) {
    return _repo->load();
  }
  else {
    return false;
  }
}

bool AnnotationService::save() {
  if (_repo) {
    return _repo->save();
  }
  else {
    return false;
  }
}
