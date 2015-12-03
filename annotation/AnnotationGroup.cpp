#include "AnnotationGroup.h"
#include <iostream>

AnnotationGroup::AnnotationGroup() :
  _name(""),
  _attributes(),
  _color("#64FE2E")
{
}

AnnotationGroup::~AnnotationGroup() {
  std::cout << "Group " << this->getName() << " destroyed!" << std::endl;
}

void AnnotationGroup::setName(const std::string& name)
{
	_name = name;
}

std::string AnnotationGroup::getColor() const {
  return _color;
}

void AnnotationGroup::setColor(const std::string& color) {
  _color = color;
}

std::string AnnotationGroup::getName() const
{
	return _name;
};

void AnnotationGroup::setGroup(const std::shared_ptr<AnnotationGroup>& parent) {
  _parent = parent;
}

std::shared_ptr<AnnotationGroup> AnnotationGroup::getGroup() const {
  std::shared_ptr<AnnotationGroup> grp = _parent.lock();
  return grp;
}

std::map<std::string, std::string> AnnotationGroup::getAttributes() const {
  return _attributes;
}

void AnnotationGroup::setAttributes(std::map<std::string, std::string> attributes) {
  _attributes = attributes;
}

void AnnotationGroup::clearAttributes() {
  _attributes.clear();
}