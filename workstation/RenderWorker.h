#ifndef RENDERWORKER_H
#define RENDERWORKER_H

#include <QThread>
#include <QMutex>
#include <memory>

class MultiResolutionImage;
class FilterInterface;
class WSITileGraphicsItem;
class RenderThread;

class RenderWorker : public QThread
{
  Q_OBJECT
    
public:
  RenderWorker(RenderThread* thread);
  ~RenderWorker();
  void abort();
  void setChannel(int channel);
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
  int _channel;
  float _opacity;
  float _foregroundImageScale;

signals:
  void tileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel);

};
  

#endif // RENDERTHREAD_H