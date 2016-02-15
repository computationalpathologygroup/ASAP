#include "AnnotationGroup.h"
#include <iostream>

AnnotationGroup::AnnotationGroup() :
  AnnotationBase(),
  _attributes()
{
}

AnnotationGroup::~AnnotationGroup() {
}

std::vector<Point> AnnotationGroup::getImageBoundingBox() const {
  std::vector<Point> bbox;
  Point topLeft(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  Point bottomRight(std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

  if (_groupMembers.empty()) {
    topLeft = Point(0, 0);
    bottomRight = Point(0, 0);
  }
  else {
    for (std::vector<std::weak_ptr<AnnotationBase> >::const_iterator it = _groupMembers.begin(); it != _groupMembers.end(); ++it) {
      if (std::shared_ptr<AnnotationBase> localMember = it->lock()) {
        std::vector<Point> memberBBox = localMember->getImageBoundingBox();
        if (memberBBox[1].getX() > bottomRight.getX()) {
          bottomRight.setX(memberBBox[1].getX());
        }
        if (memberBBox[1].getY() > bottomRight.getY()) {
          bottomRight.setY(memberBBox[1].getY());
        }
        if (memberBBox[0].getX() < topLeft.getX()) {
          topLeft.setX(memberBBox[0].getX());
        }
        if (memberBBox[0].getY() < topLeft.getY()) {
          topLeft.setY(memberBBox[0].getY());
        }
      }
    }
  }

  bbox.push_back(topLeft);
  bbox.push_back(bottomRight);
  return bbox;
}

std::vector<Point> AnnotationGroup::getLocalBoundingBox() {
  Point center = this->getCenter();
  std::vector<Point> bbox = getImageBoundingBox();
  bbox[0].setX(bbox[0].getX() - center.getX());
  bbox[1].setX(bbox[1].getX() - center.getX());
  bbox[0].setY(bbox[0].getY() - center.getY());
  bbox[1].setY(bbox[1].getY() - center.getY());
  return bbox;

}

Point AnnotationGroup::getCenter() {
  Point center(0, 0);
  std::vector<Point> bbox = getImageBoundingBox();
  center.setX((bbox[0].getX() + bbox[1].getX()) / 2.);
  center.setY((bbox[0].getY() + bbox[1].getY()) / 2.);
  return center;
}

float AnnotationGroup::getArea() const {
  float area = 0;
  for (std::vector<std::weak_ptr<AnnotationBase> >::const_iterator it = _groupMembers.begin(); it != _groupMembers.end(); ++it) {
    if (std::shared_ptr<AnnotationBase> localMember = it->lock()) {
      area += localMember->getArea();
    }
  }
  return area;
}

unsigned int AnnotationGroup::getNumberOfPoints() const {
  unsigned int nrPoints = 0;
  for (std::vector<std::weak_ptr<AnnotationBase> >::const_iterator it = _groupMembers.begin(); it != _groupMembers.end(); ++it) {
    if (std::shared_ptr<AnnotationBase> localMember = it->lock()) {
      nrPoints += localMember->getNumberOfPoints();
    }
  }
  return nrPoints;
}

std::map<std::string, std::string> AnnotationGroup::getAttributes() const {
  return _attributes;
}

void AnnotationGroup::setAttributes(std::map<std::string, std::string> attributes) {
  _attributes = attributes;
  _modified = true;
}

void AnnotationGroup::clearAttributes() {
  _attributes.clear();
}

void AnnotationGroup::addMember(std::shared_ptr<AnnotationBase> member) {
  if (member->getGroup().get() != this) {
    member->setGroup(std::dynamic_pointer_cast<AnnotationGroup>(shared_from_this()));
  }
  _groupMembers.push_back(member);
  _modified = true;
}

void AnnotationGroup::removeMember(std::shared_ptr<AnnotationBase> const member) {
  if (member->getGroup().get() == this) {
    member->setGroup(std::shared_ptr<AnnotationGroup>());
  }
  for (std::vector<std::weak_ptr<AnnotationBase> >::iterator it = _groupMembers.begin(); it != _groupMembers.end(); ++it) {
    if (it->lock() == member) {
      it = _groupMembers.erase(it);
      break;
    }
  }
  _modified = true;
}
