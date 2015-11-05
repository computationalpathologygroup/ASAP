#include "AnnotationList.h"
#include "Annotation.h"
#include "AnnotationGroup.h"

AnnotationList::AnnotationList() {
}

AnnotationList::~AnnotationList() {
  removeAllAnnotations();
  removeAllGroups();
}

bool AnnotationList::addGroup(AnnotationGroup* group) {
  if (group) {
    if (!this->getGroup(group->getName())) {
      _groups.push_back(group);
      return true;
    }
  }
  return false;
}

bool AnnotationList::addAnnotation(Annotation* annotation) {
  if (annotation) {
    if (!this->getAnnotation(annotation->getName())) {
      _annotations.push_back(annotation);
      return true;
    }
  }
  return false;
}

AnnotationGroup* AnnotationList::getGroup(const int& index) {
  if (index < 0) {
    return *(_groups.end() - abs(index));
  }
  else {
    return *(_groups.begin() + index);
  }
}

AnnotationGroup* AnnotationList::getGroup(std::string name) {
  for (std::vector<AnnotationGroup*>::const_iterator it = _groups.begin(); it != _groups.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      return (*it);
    }
  }
  return NULL;
}

Annotation* AnnotationList::getAnnotation(const int& index) {
  if (index < 0) {
    return *(_annotations.end() - abs(index));
  }
  else {
    return *(_annotations.begin() + index);
  }
}

Annotation* AnnotationList::getAnnotation(std::string name) {
  for (std::vector<Annotation*>::const_iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      return (*it);
    }
  }
  return NULL;
}

std::vector<Annotation*> AnnotationList::getAnnotations() const {
  return _annotations;
}

std::vector<AnnotationGroup*> AnnotationList::getGroups() const {
  return _groups;
}

void AnnotationList::setAnnotations(std::vector<Annotation*> annotations) {
  this->removeAllAnnotations();
  _annotations = annotations;
}

void AnnotationList::setGroups(std::vector<AnnotationGroup*> groups) {
  this->removeAllGroups();
  _groups = groups;
}

void AnnotationList::removeGroup(const int& index) {
  if (index < 0) {
    delete *(_groups.end() - abs(index));
    _groups.erase(_groups.end() - abs(index));
  }
  else {
    delete *(_groups.begin() + index);
    _groups.erase(_groups.begin() + index);
  }
}

void AnnotationList::removeGroup(std::string name) {
  for (std::vector<AnnotationGroup*>::iterator it = _groups.begin(); it != _groups.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      delete (*it);
      _groups.erase(it);
      break;
    }
  }
}

void AnnotationList::removeAnnotation(const int& index) {
  if (index < 0) {
    delete *(_annotations.end() - abs(index));
    _annotations.erase(_annotations.end() - abs(index));
  }
  else {
    delete *(_annotations.begin() + index);
    _annotations.erase(_annotations.begin() + index);
  }
}

void AnnotationList::removeAnnotation(std::string name) {
  for (std::vector<Annotation*>::iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    if ((*it) && (*it)->getName() == name) {
      delete (*it);
      _annotations.erase(it);
      break;
    }
  }
}

void AnnotationList::removeAllAnnotations() {
  for (std::vector<Annotation*>::const_iterator it = _annotations.begin(); it != _annotations.end(); ++it) {
    delete (*it);
  }
  _annotations.clear();
}

void AnnotationList::removeAllGroups() {
  for (std::vector<AnnotationGroup*>::const_iterator it = _groups.begin(); it != _groups.end(); ++it) {
    delete (*it);
  }
  _groups.clear();
}
