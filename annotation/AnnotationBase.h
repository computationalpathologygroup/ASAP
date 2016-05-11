#ifndef ANNOTATIONBASE_H
#define ANNOTATIONBASE_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "core/Point.h"
#include "annotation_export.h"

class AnnotationGroup;

class ANNOTATION_EXPORT AnnotationBase : public std::enable_shared_from_this<AnnotationBase> {
public:

  AnnotationBase();
  virtual ~AnnotationBase();

	void setName(const std::string& name);
  std::string getName() const;

  void setGroup(const std::shared_ptr<AnnotationGroup>& group);
  std::shared_ptr<AnnotationGroup> getGroup() const;

  std::string getColor() const;
  void setColor(const std::string& color);

  virtual std::vector<Point> getImageBoundingBox() const = 0;
  virtual std::vector<Point> getLocalBoundingBox() = 0;
  virtual Point getCenter() = 0;

  virtual float getArea() const = 0;
  virtual unsigned int getNumberOfPoints() const = 0;

  inline bool isModified() { return _modified; }
  inline void resetModifiedStatus() { _modified = false; }

protected:
  bool _modified;
  std::string _name;
  std::weak_ptr<AnnotationGroup> _group;
  std::string _color;
};
#endif