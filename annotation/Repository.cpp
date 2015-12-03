#include "Repository.h"
#include <string>
#include <vector>

Repository::Repository(const std::shared_ptr<AnnotationList>& list) :
  _list(list),
  _source("")
{}

Repository::~Repository() {
}

void Repository::setSource(const std::string& sourcePath) 
{
	_source = sourcePath;
}