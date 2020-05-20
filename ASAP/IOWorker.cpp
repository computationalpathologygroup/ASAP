#include "IOWorker.h" 
#include "IOThread.h"
#include "UtilityFunctions.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include <QImage>
#include <QPixmap>
#include <QPainter>

using namespace pathology;

IOWorker::IOWorker(IOThread* thread) :
  QThread(thread),
  _abort(false),
  _backgroundChannel(0),
  _foregroundChannel(0),
  _window(1.),
  _level(0.5),
  _opacity(1.0),
  _foregroundImageScale(1.),
  _LUT()
{
}

IOWorker::~IOWorker()
{
  wait();
}

void IOWorker::abort() {
  _abort = true;
}

void IOWorker::setBackgroundChannel(int channel) {
  mutex.lock();
  _backgroundChannel = channel;
  mutex.unlock();
}

void IOWorker::setForegroundChannel(int channel) {
  mutex.lock();
  _foregroundChannel = channel;
  mutex.unlock();
}

void IOWorker::setWindowAndLevel(float window, float level) {
  mutex.lock();
  _window = window;
  _level = level;
  mutex.unlock();
}

void IOWorker::setLUT(const pathology::LUT& LUT) {
  mutex.lock();
  _LUT = LUT;
  mutex.unlock();
}

void IOWorker::setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img) {
  mutex.lock();
  _bck_img = bck_img;
  mutex.unlock();
}

void IOWorker::setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale) {
  mutex.lock();
  _for_img = for_img;
  _foregroundImageScale = scale;
  mutex.unlock();
}

void IOWorker::setForegroundOpacity(const float& opacity) {
  mutex.lock();
  _opacity = opacity;
  mutex.unlock();
}

float IOWorker::getForegroundOpacity() const {
  return _opacity;
}

void IOWorker::run()
{
  forever {
    IOJob currentJob = dynamic_cast<IOThread*>(parent())->getJob();
    if (_abort) {
      return;
    }
    mutex.lock();

    std::shared_ptr<MultiResolutionImage> local_bck_img = _bck_img.lock();
    float levelDownsample = local_bck_img->getLevelDownsample(currentJob._level);
    ImageSource* foregroundTile = NULL;
    QPixmap* foregroundPixmap = NULL;
    if (std::shared_ptr<MultiResolutionImage> local_for_img = _for_img.lock()) {
      if (local_for_img->getDataType() == pathology::DataType::UChar) {
        foregroundTile = getForegroundTile<unsigned char>(local_for_img, currentJob);
        foregroundPixmap = renderForegroundImage<unsigned char>(dynamic_cast<Patch<unsigned char>*>(foregroundTile));
      }
      else if (local_for_img->getDataType() == pathology::DataType::UInt16) {
        foregroundTile = getForegroundTile<unsigned short>(local_for_img, currentJob);
        foregroundPixmap = renderForegroundImage<unsigned short>(dynamic_cast<Patch<unsigned short>*>(foregroundTile));
      }
      else if (local_for_img->getDataType() == pathology::DataType::UInt32) {
        foregroundTile = getForegroundTile<unsigned int>(local_for_img, currentJob);
        foregroundPixmap = renderForegroundImage<unsigned int>(dynamic_cast<Patch<unsigned int>*>(foregroundTile));
      }
      else if (local_for_img->getDataType() == pathology::DataType::Float) {
        foregroundTile = getForegroundTile<float>(local_for_img, currentJob);
        foregroundPixmap = renderForegroundImage<float>(dynamic_cast<Patch<float>*>(foregroundTile));
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
      emit tileLoaded(new QPixmap(temp), currentJob._imgPosX, currentJob._imgPosY, currentJob._tileSize, currentJob._tileSize*currentJob._tileSize*local_bck_img->getSamplesPerPixel(), currentJob._level, foregroundTile, foregroundPixmap);
      mutex.unlock();
    }
  }
}

template<typename T>
QPixmap IOWorker::renderBackgroundImage(std::shared_ptr<MultiResolutionImage> local_bck_img, const IOJob& currentJob, pathology::ColorType colorType) {
  float levelDownsample = local_bck_img->getLevelDownsample(currentJob._level);
  unsigned int samplesPerPixel = local_bck_img->getSamplesPerPixel();
  T *imgBuf = new T[currentJob._tileSize*currentJob._tileSize*samplesPerPixel];
  local_bck_img->getRawRegion(currentJob._imgPosX * levelDownsample * currentJob._tileSize, currentJob._imgPosY * levelDownsample * currentJob._tileSize, currentJob._tileSize, currentJob._tileSize, currentJob._level, imgBuf);
  QImage renderedImg;
  if (colorType == pathology::ColorType::RGB) {
    renderedImg = QImage(reinterpret_cast<unsigned char*>(imgBuf), (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 3, QImage::Format_RGB888);
  }
  else if (colorType == pathology::ColorType::RGBA) {
    renderedImg = QImage(reinterpret_cast<unsigned char*>(imgBuf), (currentJob._tileSize), (currentJob._tileSize), (currentJob._tileSize) * 4, QImage::Format_RGBA8888);
  }
  else {
    renderedImg = convertMonochromeToRGB(imgBuf, currentJob._tileSize, currentJob._tileSize, _backgroundChannel, samplesPerPixel, local_bck_img->getMinValue(), local_bck_img->getMaxValue(), pathology::DefaultColorLookupTables["Normal"]);
  }
  QPixmap renderedPixmap = QPixmap::fromImage(renderedImg);
  delete[] imgBuf;
  return renderedPixmap;
}

template<typename T>
Patch<T>* IOWorker::getForegroundTile(std::shared_ptr<MultiResolutionImage> local_for_img, const IOJob& currentJob) {
  std::shared_ptr<MultiResolutionImage> loc_bck_img = _bck_img.lock();
  int levelDifference = loc_bck_img->getBestLevelForDownSample(_foregroundImageScale);
  int fgImageLevel = currentJob._level - levelDifference;

  // If we request a level which is outside the range of the foreground image (e.g. level 8 when it only has 7 levels), get level 7 and scale up.
  float foregroundExtraScaling = 1.;
  if (fgImageLevel > local_for_img->getNumberOfLevels()) {
    fgImageLevel = local_for_img->getNumberOfLevels() - 1;
  }
  else if (fgImageLevel < 0) {
    fgImageLevel = 0;
  }
  foregroundExtraScaling = local_for_img->getLevelDimensions(fgImageLevel)[0] / static_cast<float>(loc_bck_img->getLevelDimensions(currentJob._level)[0]);
  pathology::DataType foregroundDataType = local_for_img->getDataType();

  int correctedTileSize = currentJob._tileSize * foregroundExtraScaling;
  unsigned int samplesPerPixel = local_for_img->getSamplesPerPixel();
  float fgLevelDownsample = local_for_img->getLevelDownsample(fgImageLevel);
  T* imgBuf = new T[correctedTileSize * correctedTileSize * samplesPerPixel];
  local_for_img->getRawRegion(currentJob._imgPosX * fgLevelDownsample * foregroundExtraScaling * currentJob._tileSize, currentJob._imgPosY * fgLevelDownsample * foregroundExtraScaling * currentJob._tileSize, correctedTileSize, correctedTileSize, fgImageLevel, imgBuf);
  Patch<T>* foregroundTile = new Patch<T>({ correctedTileSize, correctedTileSize }, local_for_img->getColorType(), imgBuf, local_for_img->getMinValues(), local_for_img->getMaxValues());
  return foregroundTile;
}

template<typename T>
QPixmap* IOWorker::renderForegroundImage(Patch<T>* foregroundTile) {
  QImage renderedImage = convertMonochromeToRGB(foregroundTile->getPointer(), correctedTileSize, correctedTileSize, _foregroundChannel, samplesPerPixel, 0, 255, _LUT);
  
  if (_foregroundImageScale != 1) {
    renderedImage = renderedImage.scaled(currentJob._tileSize, currentJob._tileSize);
  }
  QPixmap* renderedPixmap = new QPixmap::fromImage(renderedImage);
  return renderedPixmap;
}