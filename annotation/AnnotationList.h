#ifndef ANNOTATIONLIST_H
#define ANNOTATIONLIST_H

#include <string>
#include <vector>
#include <memory>
#include "annotation_export.h"

class AnnotationGroup;
class Annotation;

class ANNOTATION_EXPORT AnnotationList {
public:
  AnnotationList();
  ~AnnotationList();

  bool isModified();
  void resetModifiedStatus();
  bool addGroup(const std::shared_ptr<AnnotationGroup>& group);
  bool addAnnotation(const std::shared_ptr<Annotation>& annotation);
  std::shared_ptr<AnnotationGroup> getGroup(const int& index);
  std::shared_ptr<AnnotationGroup> getGroup(std::string name);
  std::shared_ptr<Annotation> getAnnotation(const int& index);
  std::shared_ptr<Annotation> getAnnotation(std::string name);
  std::vector<std::shared_ptr<Annotation> > getAnnotations() const;
  std::vector<std::shared_ptr<AnnotationGroup> > getGroups() const;
  void setAnnotations(const std::vector<std::shared_ptr<Annotation> >& annotations);
  void setGroups(const std::vector<std::shared_ptr<AnnotationGroup> >& groups);
  void removeGroup(const int& index);
  void removeGroup(std::string name);
  void removeAnnotation(const int& index);
  void removeAnnotation(std::string name);
  void removeAllAnnotations();
  void removeAllGroups();

private:
  std::vector<std::shared_ptr<AnnotationGroup> > _groups;
  std::vector<std::shared_ptr<Annotation> > _annotations;
};
#endif