#include "MiniMap.h"  

#include <QPixmap>
#include <QPainter>
#include <QSizePolicy>
#include <QMouseEvent>

MiniMap::MiniMap(QPixmap* overview, QWidget *parent) 
  : QWidget(parent),
    _overview(overview),
    _fieldOfView(QRectF()),
    _aspectRatio(1)
{
  //setAttribute(Qt::WA_StaticContents);
  QSizePolicy policy;
  policy.setHeightForWidth(true);
  policy.setHorizontalPolicy(QSizePolicy::Maximum);
  policy.setVerticalPolicy(QSizePolicy::Maximum);
  setSizePolicy(policy);
  if (_overview) {
    _aspectRatio = static_cast<float>(_overview->width()) / _overview->height();
  }
}

MiniMap::~MiniMap() {
  if (_overview) {
    delete _overview;
  }
}

void MiniMap::updateFieldOfView(const QRectF& fieldOfView) {
  _fieldOfView = fieldOfView;
  this->update();
}

void MiniMap::mousePressEvent(QMouseEvent *event) {
  float posX = event->pos().x();
  float posY = event->pos().y();
  QPointF pos((_overview->width() * posX) / width() + 1, (_overview->height() * posY) / height() + 1);
  emit positionClicked(pos);
}

void MiniMap::mouseMoveEvent(QMouseEvent *event) {
  // Implement later, should pan like main image
}

void MiniMap::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  if (_overview) {
    painter.drawPixmap(1, 1, width(), height(), *_overview);
    painter.setPen(QPen(Qt::white, 2));
    painter.drawRect(1, 1, width() - 2, height() - 2);
    painter.setPen(QPen(Qt::black, 1));
    painter.drawRect(0, 0, width() - 1, height() - 1);
    if (_fieldOfView.isValid() && !_fieldOfView.isEmpty()) {
      QPen blue = QPen(QColor("blue"));
      blue.setWidth(3);
      painter.setPen(blue);
      painter.drawRect(width() * (_fieldOfView.left() / _overview->width()) + 1, height() * (_fieldOfView.top() / _overview->height()) + 1,
                       width() * (_fieldOfView.width() / _overview->width()) - 2, height() * (_fieldOfView.height() / _overview->height()) - 2);
    }
  }
}

QSize MiniMap::sizeHint() const {
  QSize size(0, 0);
  unsigned int baseSize = 250;
  if (_overview) {    
    if (_overview->width() > _overview->height()) {
      size = QSize(baseSize, baseSize / _aspectRatio);
    }
    else {
      size = QSize(baseSize * _aspectRatio, baseSize);
    }
  }
  return size;
}

int MiniMap::heightForWidth(int w) const {
  return w/_aspectRatio;
}