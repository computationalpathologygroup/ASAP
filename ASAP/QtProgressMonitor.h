#ifndef QTPROGRESSMONITOR_H
#define QTPROGRESSMONITOR_H

#include "ASAP_export.h"
#include "core/ProgressMonitor.h"
#include <QObject>

class ASAP_EXPORT QtProgressMonitor : public QObject, public ProgressMonitor {
  Q_OBJECT

public :

  void setProgress(const unsigned int& progress);
  void setStatus(const std::string& status);

  signals:
    void progressChanged(int);
    void statusChanged(std::string);

};

#endif 