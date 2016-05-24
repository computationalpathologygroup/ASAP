#include "ScaleBar.h"  

#include <QPainter>
#include <QSizePolicy>

ScaleBar::ScaleBar(float resolution, QWidget *parent)
  : QWidget(parent),
   _resolution(resolution),
   _currentWidth(300)
{
  QSizePolicy policy;
  policy.setHeightForWidth(true);
  policy.setHorizontalPolicy(QSizePolicy::Fixed);
  policy.setVerticalPolicy(QSizePolicy::Fixed);
  setSizePolicy(policy);
}

ScaleBar::~ScaleBar() {
}

void ScaleBar::updateForFieldOfView(const QRectF& fieldOfView) {
  QWidget* parent = this->parentWidget();
  if (parent) {
    float parentWidth = parent->width();
    float FOVWidth = fieldOfView.width();
    QString label;
    float imagePixelsPerScreenPixel = FOVWidth / parentWidth;
    if (this->_resolution > 0) {
      float micronsPerScreenPixel = imagePixelsPerScreenPixel * this->_resolution;
      unsigned int increment = 1000;
      for (int i = 10000; i > 0; i -= increment) {
        if (i <= 1000) {
          increment = 250;
        }
        if (i <= 250) {
          increment = 25;
        }
        if (i / micronsPerScreenPixel < 300) {
          _currentWidth = i / micronsPerScreenPixel;
          updateGeometry();
          if (i >= 1000) {
            _currentLabel = QString::number(i / 1000) + " mm";
          }
          else {
            _currentLabel = QString::number(i) + " um";
          }
          break;
        }
      }
    }
    else {
      unsigned int increment = 1000;
      for (int i = 10000; i > 0; i -= increment) {
        if (i <= 1000) {
          increment = 250;
        }
        if (i <= 250) {
          increment = 25;
        }
        if (i / imagePixelsPerScreenPixel < 300) {
          _currentWidth = i / imagePixelsPerScreenPixel;
          updateGeometry();
          _currentLabel = QString::number(i) + " pixels";
          break;
        }
      }
    }
  }
}

void ScaleBar::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setPen(QPen(QColor("black")));
  painter.setBrush(QColor(255, 255, 255, 80));
  QRectF bar = QRectF(0, 0, _currentWidth, 20);
  painter.drawRect(bar);
  painter.drawText(bar, Qt::AlignCenter, _currentLabel);
}

QSize ScaleBar::sizeHint() const {
  QSize size(_currentWidth + 1, 21);
  return size;
}

int ScaleBar::heightForWidth(int w) const {
  return 21;
}