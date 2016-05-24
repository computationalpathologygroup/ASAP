#include "FilterThread.h" 

#include <limits> 

#include <QPixmap>
#include <QPen>
#include <QBrush>
#include <QGraphicsPixmapItem>
#include <QGraphicsItemGroup>
#include <QElapsedTimer>
#include <QDebug>

#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "interfaces/interfaces.h"
#include "core/PathologyEnums.h"
#include "core/Patch.h"
#include "core/Point.h"
#include "UtilityFunctions.h"

using namespace pathology;
Q_DECLARE_METATYPE(Patch<double>*)
Q_DECLARE_METATYPE(std::vector<Point>)
Q_DECLARE_METATYPE(std::vector<std::vector<Point> >)

FilterThread::FilterThread(QObject *parent) :
  QThread(parent),
  _restart(false),
  _abort(false),
  _FOV(QRectF()),
  _level(0),
  _channel(0),
  _filterPlugin(NULL)
{
}

FilterThread::~FilterThread()
{
  stopFilter();
  _mutex.lock();

  _abort = true;
  _mutex.unlock();
  while (isRunning()) {
    _condition.wakeOne();
  }

  wait();
}

void FilterThread::updateFilterResult() {
  stopFilter();
  _restart = true;
  _condition.wakeOne();
}

void FilterThread::updateFilterResult(const QRectF& FOV, std::weak_ptr<MultiResolutionImage> img, const unsigned int level, int channel)
{
  stopFilter();

  _img = img;
  _level = level;
  _FOV = FOV;
  if (channel >= 0) {
    _channel = channel;
  }

  if (!isRunning()) {
    start(HighPriority);
  }
  else {
    _restart = true;
    _condition.wakeOne();
  }
}

void FilterThread::setFilter(const std::shared_ptr<ImageFilterPluginInterface>& filterPlugin) {
  QMutexLocker locker(&_mutex);
  _filterPlugin.reset();
  _filterPlugin = filterPlugin;
}

void FilterThread::removeFilter() {
  stopFilter();
  QMutexLocker locker(&_mutex);
  _filterPlugin.reset();
}

void FilterThread::stopFilter() {
  if (_filterPlugin) {
    _filterPlugin->cancel();
  }
}

void FilterThread::run()
{
  forever{
    _mutex.lock();
    QRectF FOV = _FOV;
    int level = _level;
    std::shared_ptr<MultiResolutionImage> img = _img.lock();
    int channel = _channel;

    if (img) {
      if (_abort) {
        return;
      }
      if (_filterPlugin)
      {
        float downsample = img->getLevelDownsample(level);
        unsigned int width = FOV.width() / downsample;
        unsigned int height = FOV.height() / downsample;
        Patch<double> input = img->getPatch<double>(FOV.left(), FOV.top(), width, height, level);
        QVariant variant;
        if (!_restart) {
          _filterPlugin->filter(input, variant);
        }

        if (variant.isValid() && !variant.isNull()) {
          Patch<double>* output = variant.value<Patch<double>*>();
          // Image
          if (output && !output->empty()) {

            std::vector<unsigned long long> dims = output->getDimensions();
            unsigned char* imgBuf = new unsigned char[dims[0] * dims[1] * dims[2]];
            double *typedOutput = output->getPointer();

            QImage renderedImg;
            if (output->getColorType() == pathology::ColorType::Monochrome || output->getColorType() == pathology::ColorType::Indexed) {
              renderedImg = convertMonochromeToRGB(typedOutput, dims[0], dims[1], 0, dims[2], output->getMinValue(), output->getMaxValue(), "Normal");
            }
            else {
              std::copy(typedOutput, typedOutput + dims[0] * dims[1] * dims[2], imgBuf);
              if (output->getColorType() == pathology::ColorType::RGB) {
                renderedImg = QImage(imgBuf, dims[0], dims[1], dims[0] * dims[2], QImage::Format_RGB888);
              }
              else {
                renderedImg = QImage(imgBuf, dims[0], dims[1], dims[0] * dims[2], QImage::Format_ARGB32);
              }
            }
            QPixmap pixmap = QPixmap::fromImage(renderedImg);
            QGraphicsPixmapItem* result = new QGraphicsPixmapItem(pixmap);
            QRectF size(0, 0, width, height);
            emit filterResult(result, size);
            if (imgBuf) {
              delete[] imgBuf;
              imgBuf = NULL;
            }
          }
          else {
            // PointSet
            std::vector<Point> pts = variant.value<std::vector<Point> >();
            if (!pts.empty()) {
              QGraphicsItemGroup* result = new QGraphicsItemGroup();
              for (std::vector<Point>::const_iterator it = pts.begin(); it != pts.end(); ++it) {
                QGraphicsEllipseItem *itm = new QGraphicsEllipseItem(it->getX() - 2.5, it->getY() - 2.5, 5, 5);
                itm->setPen(QPen(QBrush("yellow"), 0));
                itm->setBrush(QBrush("yellow"));
                result->addToGroup(itm);
              }
              QRectF size(0, 0, width, height);
              emit filterResult(result, size);
            }
            else {
              // Polygons
              std::vector<std::vector<Point> > polys = variant.value<std::vector<std::vector<Point> > >();
              QGraphicsItemGroup* result = new QGraphicsItemGroup();
              for (std::vector<std::vector<Point> >::const_iterator it = polys.begin(); it != polys.end(); ++it) {
                QPolygonF poly;
                for (std::vector<Point>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2) {
                  poly << QPointF(it2->getX(), it2->getY());
                }
                QGraphicsPolygonItem *itm = new QGraphicsPolygonItem(poly);
                itm->setPen(QPen(QBrush("yellow"), 2.));
                result->addToGroup(itm);
              }
              QRectF size(0, 0, width, height);
              emit filterResult(result, size);
            }
          }
          if (output) {
            delete output;
          }
        }
      }
      if (!_restart || !_filterPlugin) {
        _condition.wait(&_mutex);
      }
      _restart = false;
      _mutex.unlock();
      if (_abort) {
        return;
      }
    }
  }
}