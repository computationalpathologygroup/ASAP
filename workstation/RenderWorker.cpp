#include "RenderWorker.h" 
#include "RenderThread.h"
#include "UtilityFunctions.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include <QImage>
#include <QPixmap>
#include <QPainter>

using namespace pathology;

RenderWorker::RenderWorker(RenderThread* thread) :
  QThread(thread),
  _abort(false),
  _backgroundChannel(0),
  _foregroundChannel(0),
  _window(1.),
  _level(0.5),
  _opacity(1.0),
  _foregroundImageScale(1.),
  _LUTname("Normal")
{
}

RenderWorker::~RenderWorker()
{
  wait();
}

void RenderWorker::abort() {
  _abort = true;
}

void RenderWorker::setBackgroundChannel(int channel) {
  mutex.lock();
  _backgroundChannel = channel;
  mutex.unlock();
}

void RenderWorker::setForegroundChannel(int channel) {
  mutex.lock();
  _foregroundChannel = channel;
  mutex.unlock();
}

void RenderWorker::setWindowAndLevel(float window, float level) {
  mutex.lock();
  _window = window;
  _level = level;
  mutex.unlock();
}

void RenderWorker::setLUT(const std::string& LUTname) {
  mutex.lock();
  _LUTname = LUTname;
  mutex.unlock();
}

void RenderWorker::setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img) {
  mutex.lock();
  _bck_img = bck_img;
  mutex.unlock();
}

void RenderWorker::setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale) {
  mutex.lock();
  _for_img = for_img;
  _foregroundImageScale = scale;
  mutex.unlock();
}

void RenderWorker::setForegroundOpacity(const float& opacity) {
  mutex.lock();
  _opacity = opacity;
  mutex.unlock();
}

float RenderWorker::getForegroundOpacity() const {
  return _opacity;
}

void RenderWorker::run()
{
  forever {
    RenderJob currentJob = dynamic_cast<RenderThread*>(parent())->getJob();
    if (_abort) {
      return;
    }
    mutex.lock();

    std::shared_ptr<MultiResolutionImage> local_bck_img = _bck_img.lock();
    float levelDownsample = local_bck_img->getLevelDownsample(currentJob._level);
    QPixmap _foreground;
    if (std::shared_ptr<MultiResolutionImage> local_for_img = _for_img.lock()) {
      if (local_for_img->getDataType() == pathology::DataType::UChar) {
        _foreground = renderForegroundImage<unsigned char>(local_for_img, currentJob, local_for_img->getColorType());
      }
      else if (local_for_img->getDataType() == pathology::DataType::UInt32) {
        _foreground = renderForegroundImage<unsigned int>(local_for_img, currentJob, local_for_img->getColorType());
      }
      else if (local_for_img->getDataType() == pathology::DataType::Float) {
        _foreground = renderForegroundImage<float>(local_for_img, currentJob, local_for_img->getColorType());
      }
    }

    if (local_bck_img) {
      QPixmap temp;
      pathology::ColorType cType = local_bck_img->getColorType();
      if (local_bck_img->getDataType() == pathology::DataType::UChar) {
        temp = renderBackgroundImage<unsigned char>(local_bck_img, currentJob, cType);
      }
      else if (local_bck_img->getDataType() == pathology::DataType::Float) {
        temp = renderBackgroundImage<float>(local_bck_img, currentJob, cType);
      }
      else if (local_bck_img->getDataType() == pathology::DataType::UInt16) {
        temp = renderBackgroundImage<unsigned short>(local_bck_img, currentJob, cType);
      }
      else if (local_bck_img->getDataType() == pathology::DataType::UInt32) {
        temp = renderBackgroundImage<unsigned int>(local_bck_img, currentJob, cType);
      }
      if (!_foreground.isNull()) {
        QPainter painter(&temp);
        painter.setOpacity(_opacity);
        painter.drawPixmap(0, 0, _foreground);
      }
      emit tileLoaded(new QPixmap(temp), currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize*currentJob._tileSize*local_bck_img->getSamplesPerPixel(), currentJob._level);
      mutex.unlock();
    }
  }
}

template<typename T>
QPixmap RenderWorker::renderBackgroundImage(std::shared_ptr<MultiResolutionImage> local_bck_img, const RenderJob& currentJob, pathology::ColorType colorType) {
  float levelDownsample = local_bck_img->getLevelDownsample(currentJob._level);
  unsigned int samplesPerPixel = local_bck_img->getSamplesPerPixel();
  T *imgBuf = new T[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
  local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
  QImage renderedImg;
  if (colorType == pathology::ColorType::RGB) {
    renderedImg = QImage(reinterpret_cast<unsigned char*>(imgBuf), (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 3, QImage::Format_RGB888);
  }
  else if (colorType == pathology::ColorType::ARGB) {
    renderedImg = QImage(reinterpret_cast<unsigned char*>(imgBuf), (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 4, QImage::Format_ARGB32);
  }
  else {
    renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _backgroundChannel, samplesPerPixel, local_bck_img->getMinValue(), local_bck_img->getMaxValue(), "Normal");
  }
  QPixmap renderedPixmap = QPixmap::fromImage(renderedImg);
  delete[] imgBuf;
  return renderedPixmap;
}

template<typename T>
QPixmap RenderWorker::renderForegroundImage(std::shared_ptr<MultiResolutionImage> local_for_img, const RenderJob& currentJob, pathology::ColorType colorType) {

  std::shared_ptr<MultiResolutionImage> loc_bck_img = _bck_img.lock();
  int levelDifference = loc_bck_img->getBestLevelForDownSample(_foregroundImageScale);
  int fgImageLevel = currentJob._level - levelDifference;

  // If we request a level which is outside the range of the foreground image (e.g. level 8 when it only has 7 levels), get level 7 and scale up.
  float foregroundExtraScaling = 1.;
  if (fgImageLevel > local_for_img->getNumberOfLevels()) {
    fgImageLevel = local_for_img->getNumberOfLevels() - 1;
    foregroundExtraScaling = local_for_img->getLevelDimensions(fgImageLevel)[0] / static_cast<float>(loc_bck_img->getLevelDimensions(currentJob._level)[0]);
  }

  unsigned int samplesPerPixel = local_for_img->getSamplesPerPixel();
  float fgLevelDownsample = local_for_img->getLevelDownsample(fgImageLevel);
  T *imgBuf = new T[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
  local_for_img->getRawRegion(currentJob._imgPosX * fgLevelDownsample * currentJob._tileSize, currentJob._imgPosY * fgLevelDownsample * currentJob._tileSize, currentJob._tileSize / foregroundExtraScaling, currentJob._tileSize / foregroundExtraScaling, fgImageLevel, imgBuf);
  QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / foregroundExtraScaling, currentJob._tileSize / foregroundExtraScaling, _foregroundChannel, samplesPerPixel, _level - _window / 2, _level + _window / 2, _LUTname);

  if (_foregroundImageScale != 1) {
    renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
  }
  QPixmap renderedPixmap = QPixmap::fromImage(renderedImage);
  delete[] imgBuf;
  return renderedPixmap;
}