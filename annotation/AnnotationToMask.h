#ifndef ANNOTATIONTOMASK_H
#define ANNOTATIONTOMASK_H

#include <string>
#include <vector>
#include <map>

#include "config/pathology_config.h"
#include "core/Point.h"

class AnnotationList;

class EXPORT_PATHOLOGYANNOTATION AnnotationToMask {

public :
  void convert(const AnnotationList* const annotationList, const std::string& maskFile, const std::vector<unsigned long long>& dimensions, const std::vector<double>& spacing, const std::map<std::string, int> nameToLabel = std::map<std::string, int>(), const std::map<std::string, int> colorToLabel = std::map<std::string, int>()) const;

private:

  inline int isLeft(Point P0, Point P1, Point P2) const
  {
    return ((P1.getX() - P0.getX()) * (P2.getY() - P0.getY())
      - (P2.getX() - P0.getX()) * (P1.getY() - P0.getY()));
  }

  int cn_PnPoly(const Point& P, const std::vector<Point>& V) const;
  int wn_PnPoly(const Point& P, const std::vector<Point>& V) const;
};

#endif