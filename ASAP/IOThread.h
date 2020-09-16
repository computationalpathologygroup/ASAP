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
class IOWorker;

class ThreadJob {
public: 
  unsigned int _tileSize;
  long long _imgPosX;
  long long _imgPosY;
  unsigned int _level;
  
  ThreadJob(unsigned int tileSize, long long imgPosX, long long imgPosY, unsigned int level) :
    _tileSize(tileSize), _imgPosX(imgPosX), _imgPosY(imgPosY), _level(level)
  {
  }

  virtual ~ThreadJob()
  {}

};

class IOJob : public ThreadJob {
public:
  IOJob(unsigned int tileSize, long long imgPosX, long long imgPosY, unsigned int level) :
    ThreadJob(tileSize, imgPosX, imgPosY, level)
  {
  }

};

class RenderJob : public ThreadJob {
public:
  ImageSource* _foregroundTile;

  RenderJob(unsigned int tileSize, long long imgPosX, long long imgPosY, unsigned int level, ImageSource* foregroundTile) :
    ThreadJob(tileSize, imgPosX, imgPosY, level),
    _foregroundTile(foregroundTile)
  {
  }

};

class IOThread : public QObject
{
  Q_OBJECT
    
public:
  IOThread(QObject *parent, unsigned int nrThreads = 2);
  ~IOThread();

  void addJob(const unsigned int tileSize, const long long imgPosX, const long long imgPosY, const unsigned int level, ImageSource* foregroundTile = NULL);
  void setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img);
  void setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale = 1.);
  
  ThreadJob* getJob();
  void clearJobs();
  unsigned int numberOfJobs();
  void shutdown();

  std::vector<IOWorker*> getWorkers();
  unsigned int getWaitingThreads();

  public slots:

  void onBackgroundChannelChanged(int channel);
  void onForegroundChannelChanged(int channel);
  void onLUTChanged(const pathology::LUT& LUTname);

protected :
  void run();

private :
  bool _abort;
  QMutex _jobListMutex;
  QWaitCondition _condition;
  std::weak_ptr<MultiResolutionImage> _bck_img;
  std::weak_ptr<MultiResolutionImage> _for_img;
  std::list<ThreadJob*> _jobList;
  std::vector<IOWorker*> _workers;
  unsigned int _threadsWaiting;
};
  

#endif // RENDERTHREAD_H