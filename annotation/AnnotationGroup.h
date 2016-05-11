#ifndef ANNOTATIONGROUP_H
#define ANNOTATIONGROUP_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "core/stringconversion.h"
#include "annotation_export.h"
#include "AnnotationBase.h"

class ANNOTATION_EXPORT AnnotationGroup : public AnnotationBase {
public:

  AnnotationGroup();
  ~AnnotationGroup();

  std::vector<Point> getImageBoundingBox() const;
  std::vector<Point> getLocalBoundingBox();
  Point getCenter();

  float getArea() const;
  unsigned int getNumberOfPoints() const;

  void clearAttributes();

  template<typename T>
  void setAttribute(const std::string& key, const T& value) {
    _attributes[key] = core::tostring<T>(value);
    _modified = true;
  }

  template<typename T>
  void setAttribute(const std::string& key, const std::vector<T>& value) {
    _attributes[key] = core::tostring<T>(value, ";");
    _modified = true;
  }

  template<typename T>
  T getAttribute(const std::string& key) {
    return core::fromstring<T>(_attributes["key"]);
  }

  template<typename T>
  std::vector<T> getAttribute(const std::string& key) {
    return core::fromstring<T>(_attributes["key"], ";");
  }

  std::map<std::string, std::string> getAttributes() const;

  void setAttributes(std::map<std::string, std::string> attributes);
  void addMember(std::shared_ptr<AnnotationBase> member);
  void removeMember(std::shared_ptr<AnnotationBase> member);

private:
  std::map<std::string, std::string> _attributes;
  std::vector<std::weak_ptr<AnnotationBase> > _groupMembers;
};
#endif