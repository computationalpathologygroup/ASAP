#ifndef ANNOTATIONTOMASK_H
#define ANNOTATIONTOMASK_H

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "annotation_export.h"
#include "core/Point.h"

class AnnotationList;
class ProgressMonitor;

class ANNOTATION_EXPORT AnnotationToMask {

public :
  void convert(const std::shared_ptr<AnnotationList>& annotationList, const std::string& maskFile, const std::vector<unsigned long long>& dimensions, const std::vector<double>& spacing, const std::map<std::string, int> nameToLabel = std::map<std::string, int>(), const std::vector<std::string> nameOrder = std::vector<std::string>()) const;
  void setProgressMonitor(ProgressMonitor* monitor);

private:

  inline int isLeft(Point P0, Point P1, Point P2) const
  {
    return static_cast<int>((P1.getX() - P0.getX()) * (P0.getY() - P2.getY())
      - (P2.getX() - P0.getX()) * (P0.getY() - P1.getY()));
  }

  int cn_PnPoly(const Point& P, const std::vector<Point>& V) const;
  int wn_PnPoly(const Point& P, const std::vector<Point>& V) const;

  ProgressMonitor* _monitor;
};

#endif