#include "AnnotationList.h"
#include "Annotation.h"
#include "AnnotationGroup.h"

AnnotationList::AnnotationList() {
}

AnnotationList::~AnnotationList() {
  removeAllAnnotations();
  removeAllGroups();
}

bool AnnotationList::isModified() {
  for (std::vector<std::shared_ptr<Annotation> >::iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    if ((*it)->isModified()) {
      return true;
    }
  }
  for (std::vector<std::shared_ptr<AnnotationGroup> >::iterator it = _groups.begin(); it != _groups.end(); ++it) {
    if ((*it)->isModified()) {
      return true;
    }
  }
  return false;
}

void AnnotationList::resetModifiedStatus() {
  for (std::vector<std::shared_ptr<Annotation> >::iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    (*it)->resetModifiedStatus();
  }
  for (std::vector<std::shared_ptr<AnnotationGroup> >::iterator it = _groups.begin(); it != _groups.end(); ++it) {
    (*it)->resetModifiedStatus();
  }
}

bool AnnotationList::addGroup(const std::shared_ptr<AnnotationGroup>& group) {
  if (group) {
    if (!this->getGroup(group->getName())) {
      _groups.push_back(group);
      return true;
    }
  }
  return false;
}

bool AnnotationList::addAnnotation(const std::shared_ptr<Annotation>& annotation) {
  if (annotation) {
    if (!this->getAnnotation(annotation->getName())) {
      _annotations.push_back(annotation);
      return true;
    }
  }
  return false;
}

std::shared_ptr<AnnotationGroup> AnnotationList::getGroup(const int& index) {
  if (index < 0) {
    return *(_groups.end() - abs(index));
  }
  else {
    return *(_groups.begin() + index);
  }
}

std::shared_ptr<AnnotationGroup> AnnotationList::getGroup(std::string name) {
  for (std::vector<std::shared_ptr<AnnotationGroup> >::const_iterator it = _groups.begin(); it != _groups.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      return (*it);
    }
  }
  return NULL;
}

std::shared_ptr<Annotation> AnnotationList::getAnnotation(const int& index) {
  if (index < 0) {
    return *(_annotations.end() - abs(index));
  }
  else {
    return *(_annotations.begin() + index);
  }
}

std::shared_ptr<Annotation> AnnotationList::getAnnotation(std::string name) {
  for (std::vector<std::shared_ptr<Annotation> >::const_iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      return (*it);
    }
  }
  return NULL;
}

std::vector<std::shared_ptr<Annotation> > AnnotationList::getAnnotations() const {
  return _annotations;
}

std::vector<std::shared_ptr<AnnotationGroup> > AnnotationList::getGroups() const {
  return _groups;
}

void AnnotationList::setAnnotations(const std::vector<std::shared_ptr<Annotation> >& annotations) {
  this->removeAllAnnotations();
  _annotations = annotations;
}

void AnnotationList::setGroups(const std::vector<std::shared_ptr<AnnotationGroup> >& groups) {
  this->removeAllGroups();
  _groups = groups;
}

void AnnotationList::removeGroup(const int& index) {
  if (index < 0) {
    (_groups.end() - abs(index))->reset();
    _groups.erase(_groups.end() - abs(index));
  }
  else {
    (_groups.begin() + index)->reset();
    _groups.erase(_groups.begin() + index);
  }
}

void AnnotationList::removeGroup(std::string name) {
  for (std::vector<std::shared_ptr<AnnotationGroup> >::iterator it = _groups.begin(); it != _groups.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      _groups.erase(it);
      break;
    }
  }
}

void AnnotationList::removeAnnotation(const int& index) {
  if (index < 0) {
    (_annotations.end() - abs(index))->reset();
    _annotations.erase(_annotations.end() - abs(index));
  }
  else {
    (_annotations.begin() + index)->reset();
    _annotations.erase(_annotations.begin() + index);
  }
}

void AnnotationList::removeAnnotation(std::string name) {
  for (std::vector<std::shared_ptr<Annotation> >::iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      _annotations.erase(it);
      break;
    }
  }
}

void AnnotationList::removeAllAnnotations() {
  _annotations.clear();
}

void AnnotationList::removeAllGroups() {
  _groups.clear();
}
