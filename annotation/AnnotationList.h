#ifndef ANNOTATIONLIST_H
#define ANNOTATIONLIST_H

#include <string>
#include <vector>
#include "config/pathology_config.h"

class AnnotationGroup;
class Annotation;

class EXPORT_PATHOLOGYANNOTATION AnnotationList {
public:
  AnnotationList();
  ~AnnotationList();

  bool addGroup(AnnotationGroup* group);
  bool addAnnotation(Annotation* annotation);
  AnnotationGroup* getGroup(const int& index);
  AnnotationGroup* getGroup(std::string name);
  Annotation* getAnnotation(const int& index);
  Annotation* getAnnotation(std::string name);
  std::vector<Annotation*> getAnnotations() const;
  std::vector<AnnotationGroup*> getGroups() const;
  void setAnnotations(std::vector<Annotation*> annotations);
  void setGroups(std::vector<AnnotationGroup*> groups);
  void removeGroup(const int& index);
  void removeGroup(std::string name);
  void removeAnnotation(const int& index);
  void removeAnnotation(std::string name);
  void removeAllAnnotations();
  void removeAllGroups();

private:
  std::vector<AnnotationGroup*> _groups;
  std::vector<Annotation*> _annotations;
};
#endif