#ifndef QTANNOTATIONGROUP_H
#define QTANNOTATIONGROUP_H
#include <string>
#include <vector>
#include <memory>

#include <QObject>

#include "annotationplugin_export.h"

class AnnotationGroup;

class ANNOTATIONPLUGIN_EXPORT QtAnnotationGroup : public QObject
{
  Q_OBJECT
public:
  QtAnnotationGroup(const std::shared_ptr<AnnotationGroup>& group, QObject *parent);
  virtual ~QtAnnotationGroup() {};
  std::shared_ptr<AnnotationGroup> getAnnotationGroup() const;

protected:
  std::shared_ptr<AnnotationGroup> _annotationGroup;
};

QDataStream &operator<<(QDataStream &out, QtAnnotationGroup* const &rhs);
QDataStream &operator>>(QDataStream &in, QtAnnotationGroup* &rhs);

#endif