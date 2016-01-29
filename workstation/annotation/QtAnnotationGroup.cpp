#include "QtAnnotationGroup.h"
#include "annotation/AnnotationGroup.h"
#include <QDataStream>

QtAnnotationGroup::QtAnnotationGroup(const std::shared_ptr<AnnotationGroup>& group, QObject *parent) :
QObject(parent),
_annotationGroup(group)
{
}

std::shared_ptr<AnnotationGroup> QtAnnotationGroup::getAnnotationGroup() const {
  return _annotationGroup;
}

QDataStream &operator<<(QDataStream &out, QtAnnotationGroup* const &rhs) {
  out.writeRawData(reinterpret_cast<const char*>(&rhs), sizeof(rhs));
  return out;
}

QDataStream &operator>>(QDataStream &in, QtAnnotationGroup* &rhs) {
  in.readRawData(reinterpret_cast<char*>(&rhs), sizeof(rhs));
  return in;
}