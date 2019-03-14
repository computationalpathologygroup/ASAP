#ifndef RENDERWORKER_H
#define RENDERWORKER_H

#include <QThread>
#include <QMutex>
#include <QPixmap>
#include <memory>
#include "core/PathologyEnums.h"
#include "RenderThread.h"

class MultiResolutionImage;
class FilterInterface;
class WSITileGraphicsItem;

class RenderWorker : public QThread
{
  Q_OBJECT
    
public:
  RenderWorker(RenderThread* thread);
  ~RenderWorker();
  void abort();

  void setBackgroundChannel(int channel);
  void setForegroundChannel(int channel);
  void setWindowAndLevel(float window, float level);
  void setLUT(const std::string& LUTname);

  void setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img);
  void setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale = 1.);
  void setForegroundOpacity(const float& opacity);
  float getForegroundOpacity() const;

protected :
  void run();

private :

  QMutex mutex;
  std::weak_ptr<MultiResolutionImage> _bck_img;
  std::weak_ptr<MultiResolutionImage> _for_img;
  bool _abort;
  int _backgroundChannel;
  int _foregroundChannel;
  float _window;
  float _level;
  float _opacity;

  //! Foreground images can only be the same size or smaller than the background images, thus this value ranges from 1 to +inf
  float _foregroundImageScale;
  std::string _LUTname;

  template <typename T>
  QPixmap renderBackgroundImage(std::shared_ptr<MultiResolutionImage> local_bck_img, const RenderJob& currentJob, pathology::ColorType colorType);

  template <typename T>
  QPixmap renderForegroundImage(std::shared_ptr<MultiResolutionImage> local_for_img, const RenderJob& currentJob, pathology::ColorType colorType);

signals:
  void tileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel);

};
  

#endif // RENDERTHREAD_H