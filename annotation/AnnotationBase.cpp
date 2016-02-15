#include "AnnotationBase.h"
#include "AnnotationGroup.h"
#include "psimpl.h"
#include <limits>

AnnotationBase::AnnotationBase() :
  _name(""),
  _color("#F4FA58"),
  _modified(true)
{
}

AnnotationBase::~AnnotationBase() {
}

std::string AnnotationBase::getColor() const {
  return _color;
}

void AnnotationBase::setColor(const std::string& color) {
  _color = color;
}

void AnnotationBase::setName(const std::string& name)
{
	_name = name;
}

std::string AnnotationBase::getName() const
{
	return _name;
};

void AnnotationBase::setGroup(const std::shared_ptr<AnnotationGroup>& group) {
  std::shared_ptr<AnnotationGroup> currentGroup = _group.lock();
  if (!currentGroup || currentGroup != group) {
    _group.reset();
    if (currentGroup) {
      currentGroup->removeMember(this->shared_from_this());
    }
    _group = group;
    if (group) {
      group->addMember(this->shared_from_this());
    }
  }
}

std::shared_ptr<AnnotationGroup> AnnotationBase::getGroup() const {
  std::shared_ptr<AnnotationGroup> grp = _group.lock();
  return grp;
}