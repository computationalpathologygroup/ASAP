#ifndef PrefetchTHREAD_H
#define PrefetchTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QRect>

class MultiResolutionImage;

class PrefetchThread : public QThread
{
  Q_OBJECT
    
public:
  PrefetchThread(QObject *parent = 0);
  ~PrefetchThread();

public slots :
  void FOVChanged(MultiResolutionImage* img, const QRectF& FOV, const unsigned int level);

protected :
  void run();

private :

  bool _restart;
  bool _abort;
  QMutex _mutex;
  QWaitCondition _condition;
  QRectF _FOV;
  unsigned int _level;
  MultiResolutionImage *_img;

};
  

#endif // PrefetchTHREAD_H