#ifndef RENDERWORKER_H
#define RENDERWORKER_H

#include <QThread>
#include <QMutex>
#include <QPixmap>
#include <memory>
#include "core/PathologyEnums.h"
#include "IOThread.h"

class MultiResolutionImage;
class FilterInterface;
class WSITileGraphicsItem;

class IOWorker : public QThread
{
  Q_OBJECT
    
public:
  IOWorker(IOThread* thread);
  ~IOWorker();
  void abort();

  void setBackgroundChannel(int channel);
  void setForegroundChannel(int channel);
  void setLUT(const pathology::LUT& LUTname);

  void setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img);
  void setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale = 1.);

signals:
  void tileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel, ImageSource* foregroundTile = NULL, QPixmap* foregroundPixmap = NULL);
  void foregroundTileRendered(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileLevel);

protected :
  void run();

private :

  QMutex mutex;
  std::weak_ptr<MultiResolutionImage> _bck_img;
  std::weak_ptr<MultiResolutionImage> _for_img;
  bool _abort;
  int _backgroundChannel;
  int _foregroundChannel;

  //! Foreground images can only be the same size or smaller than the background images, thus this value ranges from 1 to +inf
  float _foregroundImageScale;
  pathology::LUT _LUT;

  bool executeIOJob(IOJob* job);
  bool executeRenderJob(RenderJob* job);

  template <typename T>
  QPixmap* renderBackgroundImage(std::shared_ptr<MultiResolutionImage> local_bck_img, const IOJob* currentJob, pathology::ColorType colorType);

  template<typename T>
  Patch<T>* getForegroundTile(std::shared_ptr<MultiResolutionImage> local_for_img, const IOJob* currentJob);
  
  template<typename T>
  QPixmap* renderForegroundImage(Patch<T>* foregroundTile, unsigned int backgroundTileSize);

};
  

#endif // RENDERTHREAD_H