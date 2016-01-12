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
  _channel(0),
  _opacity(1.0),
  _foregroundImageScale(1.)
{
}

RenderWorker::~RenderWorker()
{
  wait();
}

void RenderWorker::abort() {
  _abort = true;
}

void RenderWorker::setChannel(int channel) {
  mutex.lock();
  _channel = channel;
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
      int levelDifference = local_bck_img->getBestLevelForDownSample(_foregroundImageScale);
      int fgImageLevel = currentJob._level - levelDifference;
      float fgImageScale = 1;
      if (fgImageLevel < 0) {
        fgImageScale = pow(2,-1*fgImageLevel);
        fgImageLevel = 0;
      }
      if (fgImageLevel > local_for_img->getNumberOfLevels()) {
        int fgMaxLevelDifference = local_for_img->getNumberOfLevels() - fgImageLevel + 1;
        fgImageLevel = local_for_img->getNumberOfLevels() - 1;
        fgImageScale = 1. / pow(2, fgMaxLevelDifference);
      }
      float fgImageLevelDownsample = local_for_img->getLevelDownsample(fgImageLevel);
      if (local_for_img->getColorType() == pathology::ColorType::Monochrome) {
        if (local_for_img->getDataType() == pathology::DataType::UChar) {
          // Label map
          unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize];
          local_for_img->getRawRegion(currentJob._imgPosX * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._imgPosY * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, fgImageLevel, imgBuf);
          QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, 0, 1, 0, 255, 0);
          if (_foregroundImageScale != 1) {
            renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
          }
          _foreground = QPixmap::fromImage(renderedImage);
        }
        else if (local_for_img->getDataType() == pathology::DataType::UInt32) {
          // Label map
          unsigned int* imgBuf = new unsigned int[currentJob._tileSize*currentJob._tileSize];
          local_for_img->getRawRegion(currentJob._imgPosX * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._imgPosY * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, fgImageLevel, imgBuf);
          QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, 0, 1, 0, 255, 0);
          if (_foregroundImageScale != 1) {
            renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
          }
          _foreground = QPixmap::fromImage(renderedImage);
        }
        else if (local_for_img->getDataType() == pathology::DataType::Float) {
          //Likelihood map
          float* imgBuf = new float[currentJob._tileSize*currentJob._tileSize];
          local_for_img->getRawRegion(currentJob._imgPosX * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._imgPosY * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, fgImageLevel, imgBuf);
          float minValue = std::numeric_limits<double>::min() == local_for_img->getMinValue() ? 0 : local_for_img->getMinValue();
          float maxValue = std::numeric_limits<double>::max() == local_for_img->getMaxValue() ? 1. : local_for_img->getMaxValue();
          QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, 0, 1, minValue, maxValue, 1);
          if (_foregroundImageScale != 1) {
            renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
          }
          _foreground = QPixmap::fromImage(renderedImage);
        }
      }
    }
    if (local_bck_img) {
      int samplesPerPixel = local_bck_img->getSamplesPerPixel();
      QImage renderedImg;
      QPixmap temp;
      if (local_bck_img->getColorType() == pathology::ColorType::RGB) {
        unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
        local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
        renderedImg = QImage(imgBuf, (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 3, QImage::Format_RGB888);
        temp = QPixmap::fromImage(renderedImg.convertToFormat(QImage::Format_ARGB32_Premultiplied));
        delete[] imgBuf;
      }
      else if (local_bck_img->getColorType() == pathology::ColorType::ARGB) {
        unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
        local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
        renderedImg = QImage(imgBuf, (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 4, QImage::Format_ARGB32);
        temp = QPixmap::fromImage(renderedImg);
        delete[] imgBuf;
      }
      else if (local_bck_img->getColorType() == pathology::ColorType::Monochrome || local_bck_img->getColorType() == pathology::ColorType::Indexed) {
        if (local_bck_img->getDataType() == pathology::DataType::UChar) {
          unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, local_bck_img->getMinValue(), local_bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (local_bck_img->getDataType() == pathology::DataType::Float) {
          float *imgBuf = new float[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, local_bck_img->getMinValue(), local_bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (local_bck_img->getDataType() == pathology::DataType::UInt16) {
          unsigned short *imgBuf = new unsigned short[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, local_bck_img->getMinValue(), local_bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (local_bck_img->getDataType() == pathology::DataType::UInt32) {
          unsigned int *imgBuf = new unsigned int[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, local_bck_img->getMinValue(), local_bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
      }
      if (!_foreground.isNull()) {
        QPainter painter(&temp);
        painter.setOpacity(_opacity);
        painter.drawPixmap(0, 0, _foreground);
      }
      emit tileLoaded(new QPixmap(temp), currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize*currentJob._tileSize*samplesPerPixel, currentJob._level);
      mutex.unlock();
    }
  }
}
