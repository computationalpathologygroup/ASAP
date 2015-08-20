#include "RenderWorker.h" 
#include "RenderThread.h"
#include "UtilityFunctions.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "WSITileGraphicsItem.h"
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
  _opacity(1.0)
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

void RenderWorker::setForegroundImage(MultiResolutionImage* for_img) {
  mutex.lock();
  _for_img = for_img;
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
    if (currentJob._sender) {
      mutex.lock();
      QPixmap _foreground;
      if (_for_img) {
        if (_for_img->getColorType() == pathology::ColorType::Monochrome) {
          if (_for_img->getDataType() == pathology::DataType::UChar) {
            // Label map
            unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize];
            _for_img->getRawRegion<unsigned char>(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
            QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, 0, 1, 0, 255, 0);
            _foreground = QPixmap::fromImage(renderedImage);
          }
          else if (_for_img->getDataType() == pathology::DataType::Float) {
            //Likelihood map
            float* imgBuf = new float[currentJob._tileSize*currentJob._tileSize];
            _for_img->getRawRegion<float>(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
            QImage renderedImage = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, 0, 1, 0, 1.2, 1);
            _foreground = QPixmap::fromImage(renderedImage);
          }
        }
      }
      if (_bck_img) {
        QImage renderedImg;
        QPixmap temp;
        if (_bck_img->getColorType() == pathology::ColorType::RGB) {
          unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*currentJob._samplesPerPixel];
          _bck_img->getRawRegion<unsigned char>(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = QImage(imgBuf, (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 3, QImage::Format_RGB888);
          temp = QPixmap::fromImage(renderedImg.convertToFormat(QImage::Format_ARGB32_Premultiplied));
          delete[] imgBuf;
        }
        else if (_bck_img->getColorType() == pathology::ColorType::ARGB) {
          unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*currentJob._samplesPerPixel];
          _bck_img->getRawRegion<unsigned char>(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
          renderedImg = QImage(imgBuf, (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 4, QImage::Format_ARGB32);
          temp = QPixmap::fromImage(renderedImg);
          delete[] imgBuf;
        }
        else if (_bck_img->getColorType() == pathology::ColorType::Monochrome || _bck_img->getColorType() == pathology::ColorType::Indexed) {
          if (_bck_img->getDataType() == pathology::DataType::UChar) {
            unsigned char* imgBuf = new unsigned char[currentJob._tileSize*currentJob._tileSize*currentJob._samplesPerPixel];
            _bck_img->getRawRegion<unsigned char>(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
            renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, currentJob._samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
            temp = QPixmap::fromImage(renderedImg);
            delete[] imgBuf;
          }
          else if (_bck_img->getDataType() == pathology::DataType::Float) {
            float *imgBuf = new float[currentJob._tileSize*currentJob._tileSize*currentJob._samplesPerPixel];
            _bck_img->getRawRegion(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
            renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, currentJob._samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
            temp = QPixmap::fromImage(renderedImg);
            delete[] imgBuf;
          }
          else if (_bck_img->getDataType() == pathology::DataType::UInt16) {
            unsigned short *imgBuf = new unsigned short[currentJob._tileSize*currentJob._tileSize*currentJob._samplesPerPixel];
            _bck_img->getRawRegion(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
            renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, currentJob._samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
            temp = QPixmap::fromImage(renderedImg);
            delete[] imgBuf;
          }
          else if (_bck_img->getDataType() == pathology::DataType::UInt32) {
            unsigned int *imgBuf = new unsigned int[currentJob._tileSize*currentJob._tileSize*currentJob._samplesPerPixel];
            _bck_img->getRawRegion(currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
            renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _channel, currentJob._samplesPerPixel, _bck_img->getMinValue(), _bck_img->getMaxValue());
            temp = QPixmap::fromImage(renderedImg);
            delete[] imgBuf;
          }
        }
        if (!_foreground.isNull()) {
          QPainter painter(&temp);
          painter.setOpacity(_opacity);
          painter.drawPixmap(0, 0, _foreground);
        }
        _thread->_senderDeletionMutex.lock();
        if (currentJob._sender) {
          currentJob._sender->initializeItem(new QPixmap(temp));
        }
        _thread->_senderDeletionMutex.unlock();
        mutex.unlock();
      }
    }
  }
}
