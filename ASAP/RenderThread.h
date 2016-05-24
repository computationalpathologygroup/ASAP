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
  RenderThread(QObject *parent, unsigned int nrThreads = 2);
  ~RenderThread();

  void addJob(const unsigned int tileSize, const long long imgPosX, const long long imgPosY, const unsigned int level);
  void setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img);
  void setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale = 1.);
  
  void setForegroundOpacity(const float& opacity);
  float getForegroundOpacity() const;

  RenderJob getJob();
  void clearJobs();
  unsigned int numberOfJobs();
  void shutdown();

  std::vector<RenderWorker*> getWorkers();
  unsigned int getWaitingThreads();

  public slots:

  void onBackgroundChannelChanged(int channel);
  void onForegroundChannelChanged(int channel);
  void onWindowAndLevelChanged(float window, float level);
  void onLUTChanged(std::string LUTname);

protected :
  void run();

private :
  bool _abort;
  QMutex _jobListMutex;
  QWaitCondition _condition;
  std::weak_ptr<MultiResolutionImage> _bck_img;
  std::weak_ptr<MultiResolutionImage> _for_img;
  std::list<RenderJob> _jobList;
  std::vector<RenderWorker*> _workers;
  unsigned int _threadsWaiting;
};
  

#endif // RENDERTHREAD_H