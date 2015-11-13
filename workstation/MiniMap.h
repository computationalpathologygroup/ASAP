#ifndef MINIMAP_H
#define MINIMAP_H

#include <QWidget>

class QPixmap;
class TileManager;

class MiniMap : public QWidget {
  Q_OBJECT

public:
  MiniMap(QPixmap* overview, QWidget *parent = 0);
  ~MiniMap();

  QSize sizeHint() const;
  int heightForWidth(int w) const;
  void setTileManager(TileManager* manager);

public slots:
  void updateFieldOfView(const QRectF& fieldOfView);
  void toggleCoverageMap(bool drawCoverageMap);
  void onCoverageUpdated();

protected:
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void paintEvent(QPaintEvent *event);

private:
  QPixmap* _overview;
  QRectF _fieldOfView;
  TileManager* _manager;
  float _aspectRatio; //Width / height
  static const char* const coverageColors[];
  bool _drawCoverageMap;

signals :
  void positionClicked(QPointF position);

};

#endif