#ifndef RENDERWORKER_H
#define RENDERWORKER_H

#include <QThread>
#include <QMutex>

class MultiResolutionImage;
class FilterInterface;
class WSITileGraphicsItem;
class RenderThread;

class RenderWorker : public QThread
{
  Q_OBJECT
    
public:
  RenderWorker(RenderThread* thread, MultiResolutionImage* _bck_img, MultiResolutionImage* _for_img = NULL, QObject *parent = NULL);
  ~RenderWorker();
  void abort();
  void setChannel(int channel);
  void setForegroundImage(MultiResolutionImage* for_img, float scale = 1.);
  void setForegroundOpacity(const float& opacity);
  float getForegroundOpacity() const;

protected :
  void run();

private :
  QMutex mutex;
  MultiResolutionImage *_bck_img;
  MultiResolutionImage *_for_img;
  RenderThread *_thread;
  bool _abort;
  int _channel;
  float _opacity;
  float _foregroundImageScale;

signals:
  void tileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel);

};
  

#endif // RENDERTHREAD_H