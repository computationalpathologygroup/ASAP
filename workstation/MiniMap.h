#ifndef MINIMAP_H
#define MINIMAP_H

#include <QWidget>

class QPixmap;

class MiniMap : public QWidget {
  Q_OBJECT

public :
  MiniMap(QPixmap* overview, QWidget *parent = 0);
  ~MiniMap();

  QSize sizeHint() const;
  int heightForWidth(int w) const;

public slots:
  void updateFieldOfView(const QRectF& fieldOfView);

protected :
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void paintEvent(QPaintEvent *event);

private :
  QPixmap* _overview;
  QRectF _fieldOfView;
  float _aspectRatio; //Width / height

signals :
  void positionClicked(QPointF position);

};

#endif