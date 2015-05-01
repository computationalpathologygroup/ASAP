#ifndef ZOOMTOOL_H
#define ZOOMTOOL_H

#include "../interfaces/interfaces.h"

class ZoomTool : public  ToolPluginInterface {
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.ZoomTool/1.0")
  Q_INTERFACES(ToolPluginInterface)

public:
  ZoomTool();
  std::string name();
  void mouseMoveEvent(QMouseEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  QAction* getToolButton();

private :
  bool _zooming;
  QPoint _prevZoomPoint;
  float _accumZoom;
};

#endif