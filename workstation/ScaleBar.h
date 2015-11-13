#ifndef SCALEBAR_H
#define SCALEBAR_H

#include <QWidget>

class ScaleBar : public QWidget {
  Q_OBJECT

public:
  ScaleBar(float resolution, QWidget *parent);
  ~ScaleBar();

  QSize sizeHint() const;
  int heightForWidth(int w) const;

  public slots:
    void updateForFieldOfView(const QRectF& fieldOfView);

protected:
    void paintEvent(QPaintEvent *event);

private:
  float _resolution;
  QString _currentLabel;
  float _currentWidth;

};

#endif