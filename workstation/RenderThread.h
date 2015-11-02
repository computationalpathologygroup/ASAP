#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <QObject>
#include <QPointer>
#include <QMutex>
#include <QWaitCondition>
#include <memory>
#include <list>
#include "interfaces/interfaces.h"

class MultiResolutionImage;
class WSITileGraphicsItem;
class RenderWorker;

struct RenderJob {
  unsigned int _tileSize;
  long long _imgPosX;
  long long _imgPosY;
  unsigned int _level;
};

class RenderThread : public QObject
{
  Q_OBJECT
    
public:
  RenderThread(MultiResolutionImage* bck_img, unsigned int lastRenderLevel, MultiResolutionImage* for_img = NULL, unsigned int nrThreads = 2, QObject *parent = 0);
  ~RenderThread();

  void addJob(const unsigned int tileSize, const long long imgPosX, const long long imgPosY, const unsigned int level);
  void setForegroundImage(MultiResolutionImage* for_img);
  
  void setForegroundOpacity(const float& opacity);
  float getForegroundOpacity() const;

  RenderJob getJob();
  void clearJobs();
  unsigned int numberOfJobs();
  void shutdown();

  std::vector<RenderWorker*> getWorkers();

  public slots:

  void onChannelChanged(int channel);

protected :
  void run();

private :
  bool _abort;
  QMutex _jobListMutex;
  QWaitCondition _condition;
  MultiResolutionImage *_bck_img;
  MultiResolutionImage *_for_img;
  std::list<RenderJob> _jobList;
  std::vector<RenderWorker*> _workers;
  int _channel;
  float _opacity;
  unsigned int _lastRenderLevel;
};
  

#endif // RENDERTHREAD_H