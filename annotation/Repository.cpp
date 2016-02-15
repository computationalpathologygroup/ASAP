#include "Repository.h"
#include <string>
#include <vector>
#include "AnnotationList.h"

Repository::Repository(const std::shared_ptr<AnnotationList>& list) :
  _list(list),
  _source("")
{}

Repository::~Repository() {
}

bool Repository::load() {
  bool loadSucces = false;
  if (_list) {
    loadSucces = loadFromRepo();
    _list->resetModifiedStatus();
  }
  return loadSucces;
}

void Repository::setSource(const std::string& sourcePath) 
{
	_source = sourcePath;
}