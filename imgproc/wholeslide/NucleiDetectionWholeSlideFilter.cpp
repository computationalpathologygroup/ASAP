#include "NucleiDetectionWholeSlideFilter.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "core/PathologyEnums.h"
#include "imgproc/opencv/NucleiDetectionFilter.h"
#include "annotation/Annotation.h"
#include "annotation/AnnotationList.h"
#include "annotation/XmlRepository.h"
#include <memory>
#include <iostream>

NucleiDetectionWholeSlideFilter::NucleiDetectionWholeSlideFilter() :
_monitor(NULL),
_processedLevel(0),
_outPath(""),
_threshold(0.1),
_alpha(0.2),
_beta(0.1),
_centerPoints()
{
}

NucleiDetectionWholeSlideFilter::~NucleiDetectionWholeSlideFilter() {
}

void NucleiDetectionWholeSlideFilter::setInput(const std::shared_ptr<MultiResolutionImage>& input) {
  _input = input;
}

void NucleiDetectionWholeSlideFilter::setOutput(const std::string& outPath) {
  _outPath = outPath;
}

void NucleiDetectionWholeSlideFilter::setProcessedLevel(const unsigned int processedLevel) {
  _processedLevel = processedLevel;
}

unsigned int NucleiDetectionWholeSlideFilter::getProcessedLevel() {
  return _processedLevel;
}

void NucleiDetectionWholeSlideFilter::setProgressMonitor(ProgressMonitor* progressMonitor) {
  _monitor = progressMonitor;
}

ProgressMonitor* NucleiDetectionWholeSlideFilter::getProgressMonitor() {
  return _monitor;
}

void NucleiDetectionWholeSlideFilter::setThreshold(const float& threshold) {
  this->_threshold = threshold;
}

float NucleiDetectionWholeSlideFilter::getThreshold() {
  return _threshold;
}

float NucleiDetectionWholeSlideFilter::getAlpha() {
  return _alpha;
}

void NucleiDetectionWholeSlideFilter::setAlpha(const float& alpha) {
  _alpha = alpha;
}

float NucleiDetectionWholeSlideFilter::getBeta() {
  return _beta;
}

void NucleiDetectionWholeSlideFilter::setBeta(const float& beta) {
  _beta = beta;
}

void NucleiDetectionWholeSlideFilter::setMaximumRadius(const float& maxRadius) {
  _maxRadius = maxRadius;
}

float NucleiDetectionWholeSlideFilter::getMaximumRadius() {
  return _maxRadius;
}

void NucleiDetectionWholeSlideFilter::setMinimumRadius(const float& minRadius) {
  _minRadius = minRadius;
}

float NucleiDetectionWholeSlideFilter::getMinimumRadius() {
  return _minRadius;
}

void NucleiDetectionWholeSlideFilter::setRadiusStep(const float& stepRadius) {
  _stepRadius = stepRadius;
}

float NucleiDetectionWholeSlideFilter::getRadiusStep() {
  return _stepRadius;
}

bool NucleiDetectionWholeSlideFilter::process() {
  std::shared_ptr<MultiResolutionImage> img = _input.lock();
  std::vector<unsigned long long> dims = img->getLevelDimensions(this->_processedLevel);
  double downsample = img->getLevelDownsample(this->_processedLevel);
  _centerPoints.clear();
  NucleiDetectionFilter<double> filter;
  filter.setAlpha(_alpha);
  filter.setBeta(_beta);
  filter.setHMaximaThreshold(_threshold);
  filter.setMaximumRadius(_maxRadius);
  filter.setMinimumRadius(_minRadius);
  filter.setRadiusStep(_stepRadius);
  std::vector<Point> tmp;
  for (unsigned long long t_y = 0; t_y < dims[1]; t_y += 512) {
    std::cout << t_y << std::endl;
    for (unsigned long long t_x = 0; t_x < dims[0]; t_x += 512) {
      Patch<double> tile = img->getPatch<double>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, this->_processedLevel);
      double* buf = tile.getPointer();
      filter.filter(tile, tmp);
      for (std::vector<Point>::const_iterator it = tmp.begin(); it != tmp.end(); ++it) {
        std::vector<float> curPoint;
        curPoint.push_back(it->getX() * downsample + t_x*downsample);
        curPoint.push_back(it->getY() * downsample + t_y*downsample);
        _centerPoints.push_back(curPoint);
      }
      tmp.clear();
    }
  }
  if (!_outPath.empty()) {
    std::shared_ptr<Annotation> annot(new Annotation());
    annot->setName("Detected nuclei");
    annot->setType(Annotation::POINTSET);
    for (std::vector<std::vector<float> >::const_iterator it = _centerPoints.begin(); it != _centerPoints.end(); ++it) {
      float x = (*it)[0];
      float y = (*it)[1];
      annot->addCoordinate(Point(x, y));
    }
    std::shared_ptr<AnnotationList> annotList(new AnnotationList());
    annotList->addAnnotation(annot);
    XmlRepository repo(annotList);
    repo.setSource(_outPath);
    repo.save();
  }
  return true;
}

std::vector<std::vector<float> > NucleiDetectionWholeSlideFilter::getCenterPoints() {
  return _centerPoints;
}