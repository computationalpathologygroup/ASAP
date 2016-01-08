#include "RenderWorker.h" 
#include "RenderThread.h"
#include "UtilityFunctions.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include <QImage>
#include <QPixmap>
#include <QPainter>

using namespace pathology;

RenderWorker::RenderWorker(RenderThread* thread, MultiResolutionImage* bck_img, MultiResolutionImage* for_img, QObject *parent) :
  QThread(parent),
  _thread(thread),
  _bck_img(bck_img),
  _for_img(for_img),
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

void RenderWorker::setForegroundImage(MultiResolutionImage* for_img, float scale) {
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
    RenderJob currentJob = _thread->getJob();
    if (_abort) {
      return;
    }
    mutex.lock();
    float levelDownsample = _bck_img->getLevelDownsample(currentJob._level);
    QPixmap _foreground;
    if (_for_img) {
      int levelDifference = _bck_img->getBestLevelForDownSample(_foregroundImageScale);
      int fgImageLevel = currentJob._level - levelDifference;
      float fgImageScale = 1;
      if (fgImageLevel < 0) {
        fgImageScale = pow(2,-1*fgImageLevel);
        fgImageLevel = 0;
      }
      if (fgImageLevel > _for_img->getNumberOfLevels()) {
        int fgMaxLevelDifference = _for_img->getNumberOfLevels() - fgImageLevel + 1;
        fgImageLevel = _for_img->getNumberOfLevels() - 1;
        fgImageScale = 1. / pow(2, fgMaxLevelDifference);
      }
      float fgImageLevelDownsample = _for_img->getLevelDownsample(fgImageLevel);
      if (_for_img->getColorType() == pathology::ColorType::Monochrome) {
        if (_for_img->getDataType() == pathology::DataType::UChar) {
          // Label map
          unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize];
          _for_img->getRawRegion(currentJob._imgPosX * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._imgPosY * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, fgImageLevel, imgBuf);
          QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, 0, 1, 0, 255, 0);
          if (_foregroundImageScale != 1) {
            renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
          }
          _foreground = QPixmap::fromImage(renderedImage);
        }
        else if (_for_img->getDataType() == pathology::DataType::UInt32) {
          // Label map
          unsigned int* imgBuf = new unsigned int[currentJob._tileSize*currentJob._tileSize];
          _for_img->getRawRegion(currentJob._imgPosX * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._imgPosY * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, fgImageLevel, imgBuf);
          QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, 0, 1, 0, 255, 0);
          if (_foregroundImageScale != 1) {
            renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
          }
          _foreground = QPixmap::fromImage(renderedImage);
        }
        else if (_for_img->getDataType() == pathology::DataType::Float) {
          //Likelihood map
          float* imgBuf = new float[currentJob._tileSize*currentJob._tileSize];
          _for_img->getRawRegion(currentJob._imgPosX * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._imgPosY * fgImageLevelDownsample * currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, fgImageLevel, imgBuf);
          float minValue = std::numeric_limits<double>::min() == _for_img->getMinValue() ? 0 : _for_img->getMinValue();
          float maxValue = std::numeric_limits<double>::max() == _for_img->getMaxValue() ? 1. : _for_img->getMaxValue();
          QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize / fgImageScale, currentJob._tileSize / fgImageScale, 0, 1, minValue, maxValue, 1);
          if (_foregroundImageScale != 1) {
            renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
          }
          _foreground = QPixmap::fromImage(renderedImage);
        }
      }
    }
    if (_bck_img) {
      int samplesPerPixel = _bck_img->getSamplesPerPixel();
      QImage renderedImg;
      QPixmap temp;
      if (_bck_img->getColorType() == pathology::ColorType::RGB) {
        unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
        _bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
        renderedImg = QImage(imgBuf, (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 3, QImage::Format_RGB888);
        temp = QPixmap::fromImage(renderedImg.convertToFormat(QImage::Format_ARGB32_Premultiplied));
        delete[] imgBuf;
      }
      else if (_bck_img->getColorType() == pathology::ColorType::ARGB) {
        unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
        _bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
        renderedImg = QImage(imgBuf, (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 4, QImage::Format_ARGB32);
        temp = QPixmap::fromImage(renderedImg);
        delete[] imgBuf;
      }
      else if (_bck_img->getColorType() == pathology::ColorType::Monochrome || _bck_img->getColorType() == pathology::ColorType::Indexed) {
        if (_bck_img->getDataType() == pathology::DataType::UChar) {
          unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          _bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (_bck_img->getDataType() == pathology::DataType::Float) {
          float *imgBuf = new float[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          _bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (_bck_img->getDataType() == pathology::DataType::UInt16) {
          unsigned short *imgBuf = new unsigned short[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          _bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (_bck_img->getDataType() == pathology::DataType::UInt32) {
          unsigned int *imgBuf = new unsigned int[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
          _bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
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
