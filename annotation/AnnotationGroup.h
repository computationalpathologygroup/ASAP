#ifndef ANNOTATIONGROUP_H
#define ANNOTATIONGROUP_H

#include <string>
#include <vector>
#include <map>
#include "core/stringconversion.h"
#include "config/pathology_config.h"

class EXPORT_PATHOLOGYANNOTATION AnnotationGroup {
public:

  AnnotationGroup();

	void setName(const std::string& name);
	std::string getName() const;

  void clearAttributes();

  template<typename T>
  void setAttribute(const std::string& key, const T& value) {
    _attributes[key] = core::tostring<T>(value);
  }

  template<typename T>
  void setAttribute(const std::string& key, const std::vector<T>& value) {
    _attributes[key] = core::tostring<T>(value, ";");
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

  void setGroup(AnnotationGroup* parent);
  AnnotationGroup* getGroup() const;

  std::string getColor() const;
  void setColor(const std::string& color);

private:
  AnnotationGroup* _parent;
  std::string _name;
  std::map<std::string, std::string> _attributes;
  std::string _color;
};
#endif