#ifndef PANTOOL_H
#define PANTOOL_H

#include "../interfaces/interfaces.h"

class PanTool : public  ToolPluginInterface {
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.PanTool/1.0")
  Q_INTERFACES(ToolPluginInterface)

public :
  std::string name();
  void mouseMoveEvent(QMouseEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  QAction* getToolButton();
};

#endif