#include "MiniMap.h"  

#include <QPixmap>
#include <QPainter>
#include <QSizePolicy>
#include <QMouseEvent>
#include <QPainterPath>
#include "TileManager.h"

const char* const MiniMap::coverageColors[] = { "red", "green", "yellow", "black", "purple", "orange" };

MiniMap::MiniMap(const QPixmap& overview, QWidget *parent) 
  : QWidget(parent),
    _overview(overview),
    _fieldOfView(QRectF()),
    _aspectRatio(1),
    _manager(NULL),
    _drawCoverageMap(true)
{
  QSizePolicy policy;
  policy.setHeightForWidth(true);
  policy.setHorizontalPolicy(QSizePolicy::Fixed);
  policy.setVerticalPolicy(QSizePolicy::Fixed);
  setSizePolicy(policy);
  if (!overview.isNull()) {
    _aspectRatio = static_cast<float>(_overview.width()) / _overview.height();
  }
}

void MiniMap::toggleCoverageMap(bool drawCoverageMap) {
  bool repaintNeeded = false;
  if (_drawCoverageMap != drawCoverageMap) {
    repaintNeeded = true;
  }
  _drawCoverageMap = drawCoverageMap;
  if (repaintNeeded) {
    this->repaint();
  }
}

void MiniMap::setTileManager(TileManager* manager) {
  _manager = manager;
}

void MiniMap::updateFieldOfView(const QRectF& fieldOfView) {
  _fieldOfView = fieldOfView;
  this->update();
}

void MiniMap::onCoverageUpdated() {
  this->update();
}

void MiniMap::mousePressEvent(QMouseEvent *event) {
  float posX = event->pos().x();
  float posY = event->pos().y();
  QPointF pos((_overview.width() * posX) / width() + 1, (_overview.height() * posY) / height() + 1);
  emit positionClicked(pos);
}

void MiniMap::mouseMoveEvent(QMouseEvent *event) {
  // Implement later, should pan like main image
}

void MiniMap::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  if (!_overview.isNull()) {
    painter.drawPixmap(1, 1, width(), height(), _overview);
    painter.setPen(QPen(Qt::white, 2));
    painter.drawRect(1, 1, width() - 2, height() - 2);
    painter.setPen(QPen(Qt::black, 1));
    painter.drawRect(0, 0, width() - 1, height() - 1);
    if (_manager && _drawCoverageMap) {
      painter.save();
      std::vector<QPainterPath> pths = _manager->getCoverageMaps();
      QPainterPath bck;
      bck.addRect(QRectF(0, 0, width() - 1., height() - 1.));
      painter.setPen(Qt::PenStyle::NoPen);
      painter.setBrush(QBrush(QColor(0, 0, 0, 50)));
      painter.drawPath(bck);
      for (std::vector<QPainterPath>::const_iterator it = pths.begin(); it != pths.end(); ++it) {
        if (!it->isEmpty()) {
          QTransform trans;
          trans = trans.scale(width() / static_cast<float>(_overview.width()), height() / static_cast<float>(_overview.height()));
          QPainterPath qpf2 = trans.map(*it);
          unsigned int colorIndex = (it - pths.begin()) % 6;
          painter.setPen(QPen(QColor(coverageColors[colorIndex])));
          painter.setBrush(QBrush(QColor(255, 255, 255, 50 / (it - pths.begin() + 1))));
          painter.drawPath(qpf2);
        }
      }
      painter.restore();
    }
    if (_fieldOfView.isValid() && !_fieldOfView.isEmpty()) {
      QPen blue = QPen(QColor("blue"));
      blue.setWidth(3);
      painter.setPen(blue);
      float rectX = width() * (_fieldOfView.left() / _overview.width()) + 1;
      float rectY = height() * (_fieldOfView.top() / _overview.height()) + 1;
      float rectW = width() * (_fieldOfView.width() / _overview.width()) - 2;
      float rectH = height() * (_fieldOfView.height() / _overview.height()) - 2;
      if (rectW > 3 && rectH > 3) {
        painter.drawRect(rectX, rectY, rectW, rectH);
      }
      else {
        painter.drawLine(rectX + rectW / 2. - 5, rectY + rectH / 2., rectX + rectW / 2. + 5, rectY + rectH / 2.);
        painter.drawLine(rectX + rectW / 2., rectY + rectH / 2. - 5, rectX + rectW / 2., rectY + rectH / 2. + 5);
      }
    }
  }
}

QSize MiniMap::sizeHint() const {
  QSize size(0, 0);
  unsigned int baseSize = 250;
  if (!_overview.isNull()) {    
    if (_overview.width() > _overview.height()) {
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