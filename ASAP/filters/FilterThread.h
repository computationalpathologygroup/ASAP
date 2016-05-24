#ifndef FILTERTHREAD_H
#define FILTERTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QRect>
#include <QImage>
#include <vector>
#include <memory>

class MultiResolutionImage;
class ImageFilterPluginInterface;
class QGraphicsItem;

class FilterThread : public QThread
{
  Q_OBJECT

public:
  FilterThread(QObject *parent = 0);
  ~FilterThread();

  void setFilter(const std::shared_ptr<ImageFilterPluginInterface>& filterPlugin);
  void removeFilter();
  void stopFilter();

public slots :
  void updateFilterResult();
  void updateFilterResult(const QRectF& FOV, std::weak_ptr<MultiResolutionImage> img, const unsigned int level, int channel = -1);

signals:
  void filterResult(QGraphicsItem* result, QRectF size);

protected:
  void run();

private:
  bool _restart;
  bool _abort;
  unsigned int _channel;
  QMutex _mutex;
  QWaitCondition _condition;
  QRectF _FOV;
  unsigned int _level;
  std::weak_ptr<MultiResolutionImage> _img;

  // Pointer to Workstation-filter that needs to be applied
  std::shared_ptr<ImageFilterPluginInterface> _filterPlugin;

};


#endif // FILTERTHREAD_H