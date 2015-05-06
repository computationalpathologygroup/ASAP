#include "Repository.h"
#include <string>
#include <vector>
#include <qdebug.h>

Repository::Repository(AnnotationList* list) :
  _list(list),
  _source("")
{}

Repository::~Repository() {
}

void Repository::setSource(const std::string& sourcePath) 
{
	_source = sourcePath;
}