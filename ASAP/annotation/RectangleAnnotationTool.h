#ifndef RECTANGLEANNOTATIONTOOL_H
#define RECTANGLEANNOTATIONTOOL_H

#include "AnnotationTool.h"
#include "core/Point.h"
#include <string>
#include <QGraphicsTextItem>
#include <QGraphicsRectItem>

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;
class QGraphicsRectItem;

class ANNOTATIONPLUGIN_EXPORT RectangleAnnotationTool : public AnnotationTool {
  Q_OBJECT

public :
  RectangleAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  virtual std::string name();
  virtual QAction* getToolButton();
  void mouseMoveEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);
  void cancelAnnotation();  

private:

  class QGraphicsTextItemWithBackground : public QGraphicsTextItem
  {
  public:
    QGraphicsTextItemWithBackground(const QString &text);
    QRectF boundingRect() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *o, QWidget *w);
  };

  void addCoordinate(const QPointF& scenePos);
  QGraphicsRectItem* _activeRect;
  QGraphicsTextItemWithBackground* _sizeText;
  std::string _lastAction;
};

#endif