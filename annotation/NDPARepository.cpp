#include "NDPARepository.h"
#include "Annotation.h"
#include "AnnotationGroup.h"
#include "AnnotationList.h"
#include "MultiResolutionImageReader.h"
#include "OpenSlideImage.h"
#include "core/Point.h"
#include "core/filetools.h"
#include "core/stringconversion.h"
#include <vector>
#include <string>
#include <stdio.h>
#include "pugixml.hpp"

NDPARepository::NDPARepository(AnnotationList* list) :
  Repository(list)
{
}

bool NDPARepository::save() const
{
  return false;
}

void NDPARepository::setNDPISourceFile(const std::string& ndpiSourcefile) {
  _ndpiSourceFile = ndpiSourcefile;
}

std::string NDPARepository::NDPISourceFile() const {
  return _ndpiSourceFile;
}

bool NDPARepository::load()
{
  if (!_list || _source.empty()) {
    return false;
  }

  _list->removeAllAnnotations();
  _list->removeAllGroups();

  OpenSlideImage* ndpi = NULL;
  if (_ndpiSourceFile.empty()) {
    std::vector<std::string> ndpaParts;
    core::split(_source, ndpaParts, ".ndpa");
    if (core::fileExists(ndpaParts[0])) {
      MultiResolutionImageReader reader;
      ndpi = dynamic_cast<OpenSlideImage*>(reader.open(ndpaParts[0]));
      if (!ndpi) {
        return false;
      }
    }
    else {
      return false;
    }
  }

  float offsetX = core::fromstring<float>(ndpi->getOpenSlideProperty("hamamatsu.XOffsetFromSlideCentre"));
  float offsetY = core::fromstring<float>(ndpi->getOpenSlideProperty("hamamatsu.YOffsetFromSlideCentre"));
  float mppX = core::fromstring<float>(ndpi->getOpenSlideProperty("openslide.mpp-x"));
  float mppY = core::fromstring<float>(ndpi->getOpenSlideProperty("openslide.mpp-y"));
  std::vector<unsigned long long> dims = ndpi->getDimensions();

	pugi::xml_document xml_doc;
  pugi::xml_parse_result tree = xml_doc.load_file(_source.c_str());
  pugi::xml_node root = xml_doc.child("annotations");
  unsigned int annotation_nr = 0;
  for (pugi::xml_node it = root.child("ndpviewstate"); it; it = it.next_sibling("ndpviewstate"))
	{
    for (pugi::xml_node annotation_xml = it.child("annotation"); annotation_xml; annotation_xml = annotation_xml.next_sibling("annotation"))
    {
      if (std::string(annotation_xml.attribute("type").value()) == std::string("freehand")) {
        Annotation* annotation = new Annotation();

        annotation->setName(std::string(it.child_value("title")) + "_" + core::tostring(annotation_nr));
        annotation->setTypeFromString("Polygon");
        std::string annotColor = annotation_xml.attribute("color").value();
        if (!annotColor.empty()) {
          annotation->setColor(annotColor);
        }

        pugi::xml_node coordinates = annotation_xml.child("pointlist");
        for (pugi::xml_node_iterator cit = coordinates.begin(); cit != coordinates.end(); ++cit)
        {
          double x = core::fromstring<double>(cit->child_value("x"));
          double y = core::fromstring<double>(cit->child_value("y"));
          double corX = ((x - offsetX) / (mppX * 1000)) + (dims[0] / 2.);
          double corY = ((y - offsetY) / (mppY * 1000)) + (dims[1] / 2.);
          annotation->addCoordinate(corX, corY);
        }
        _list->addAnnotation(annotation);
        annotation_nr++;
      }
    }
	}
}