#include "ImageScopeRepository.h"
#include "Annotation.h"
#include "AnnotationGroup.h"
#include "AnnotationList.h"
#include "core/Point.h"
#include "core/filetools.h"
#include "core/stringconversion.h"
#include <vector>
#include <string>
#include <stdio.h>
#include "pugixml.hpp"

ImageScopeRepository::ImageScopeRepository(const std::shared_ptr<AnnotationList>& list) :
  Repository(list),
  _closingDistance(30.)
{
}

void ImageScopeRepository::setClosingDistance(const float& closingDistance) {
  _closingDistance = closingDistance;
}

float ImageScopeRepository::getClosingDistance() {
  return _closingDistance;
}

bool ImageScopeRepository::save() const
{
  return false;
}

bool ImageScopeRepository::loadFromRepo()
{
  if (!_list || _source.empty()) {
    return false;
  }

  _list->removeAllAnnotations();
  _list->removeAllGroups();

	pugi::xml_document xml_doc;
  pugi::xml_parse_result tree = xml_doc.load_file(_source.c_str());
  pugi::xml_node root = xml_doc.child("Annotations");
  if (root.empty()) {
    return false;
  }
  unsigned int group_nr = 0;
  unsigned int annot_nr = 0;
  for (pugi::xml_node grpIt = root.child("Annotation"); grpIt; grpIt = grpIt.next_sibling("Annotation"))
	{
    std::string groupName = grpIt.attribute("Name").value();
    unsigned int groupColorAsInt = core::fromstring<unsigned int>(std::string(grpIt.attribute("LineColor").value()));

    std::stringstream stream;
    stream << std::hex << groupColorAsInt;
    std::string groupColorAsHex(stream.str());
    while (groupColorAsHex.size() < 6) {
      groupColorAsHex = "0" + groupColorAsHex;
    }
    groupColorAsHex = "#" + groupColorAsHex;

    std::shared_ptr<AnnotationGroup> grp = std::make_shared<AnnotationGroup>();
    grp->setColor(groupColorAsHex);
    std::map<unsigned int, std::vector<std::pair<double, double> > > idToCoords;
    std::map<unsigned int, std::string> idToName;
    pugi::xml_node regions = grpIt.child("Regions");
    for (pugi::xml_node annotation_xml = regions.child("Region"); annotation_xml; annotation_xml = annotation_xml.next_sibling("Region"))
    {
      unsigned int id = core::fromstring<int>(annotation_xml.attribute("Id").value());
      idToName[id] = std::string(annotation_xml.attribute("Text").value());

      pugi::xml_node vertices = annotation_xml.child("Vertices");
      for (pugi::xml_node_iterator cit = vertices.begin(); cit != vertices.end(); ++cit)
      {
        double x = core::fromstring<double>(cit->attribute("X").value());
        double y = core::fromstring<double>(cit->attribute("Y").value());
        idToCoords[id].push_back(std::pair<double, double>(x, y));
      }
    }

    std::map<unsigned int, unsigned int> usedIds;
    for (std::map<unsigned int, std::string>::iterator idIt = idToName.begin(); idIt != idToName.end(); ++idIt) {
      usedIds[idIt->first] = 0;
    }
    // Now figure out which regions belong together
    for (std::map<unsigned int, std::vector<std::pair<double, double> > >::const_iterator coordIt = idToCoords.begin(); coordIt != idToCoords.end(); ++coordIt) {
      unsigned int curId = coordIt->first;
      if (usedIds[curId] == 1) {
        continue;
      }
      usedIds[curId] = 1;
      std::string curName = idToName[curId];
      std::vector<std::pair<double, double> > closedCoordList = coordIt->second;
      double dist = sqrt(pow(closedCoordList.begin()->first - closedCoordList.back().first, 2) + pow(closedCoordList.begin()->second - closedCoordList.back().second, 2));
      if (dist > _closingDistance) {
        bool isOpen = true;
        for (std::map<unsigned int, std::vector<std::pair<double, double> > >::const_iterator coordIt2 = idToCoords.begin(); coordIt2 != idToCoords.end(); ++coordIt2) {
          if (usedIds[coordIt2->first] == 1) {
            continue;
          }
          double distFirstFirst = sqrt(pow(closedCoordList.begin()->first - coordIt2->second.begin()->first, 2) + pow(closedCoordList.begin()->second - coordIt2->second.begin()->second, 2));
          double distFirstLast = sqrt(pow(closedCoordList.begin()->first - coordIt2->second.back().first, 2) + pow(closedCoordList.begin()->second - coordIt2->second.back().second, 2));
          double distLastFirst = sqrt(pow(closedCoordList.back().first - coordIt2->second.begin()->first, 2) + pow(closedCoordList.back().second - coordIt2->second.begin()->second, 2));
          double distLastLast = sqrt(pow(closedCoordList.back().first - coordIt2->second.back().first, 2) + pow(closedCoordList.back().second - coordIt2->second.back().first, 2));
          if (distLastFirst < _closingDistance) {
            closedCoordList.insert(closedCoordList.end(), coordIt2->second.begin(), coordIt2->second.end());
            usedIds[coordIt2->first] = 1;
            if (curName.empty()) {
              curName = idToName[coordIt2->first];
            }
            coordIt2 = idToCoords.begin();
          }
          else if (distLastLast < _closingDistance) {
            std::vector< std::pair<double, double> > reversed = coordIt2->second;
            std::reverse(reversed.begin(), reversed.end());
            closedCoordList.insert(closedCoordList.end(), reversed.begin(), reversed.end());
            usedIds[coordIt2->first] = 1;
            if (curName.empty()) {
              curName = idToName[coordIt2->first];
            }
            coordIt2 = idToCoords.begin();
          }
          else if (distFirstLast < _closingDistance) {
            closedCoordList.insert(closedCoordList.begin(), coordIt2->second.begin(), coordIt2->second.end());
            usedIds[coordIt2->first] = 1;
            if (curName.empty()) {
              curName = idToName[coordIt2->first];
            }
            coordIt2 = idToCoords.begin();
          }
          else if (distFirstFirst < _closingDistance) {
            std::vector< std::pair<double, double> > reversed = coordIt2->second;
            std::reverse(reversed.begin(), reversed.end());
            closedCoordList.insert(closedCoordList.begin(), reversed.begin(), reversed.end());
            usedIds[coordIt2->first] = 1;
            if (curName.empty()) {
              curName = idToName[coordIt2->first];
            }
            coordIt2 = idToCoords.begin();
          }
        }
      }
      if (groupName.empty() && !curName.empty()) {
        groupName = curName;
      }
      std::shared_ptr<Annotation> annot = std::make_shared<Annotation>();
      annot->setName(curName + "_" + core::tostring(annot_nr));
      annot->setTypeFromString("Polygon");
      annot_nr += 1;
      for (std::vector<std::pair<double, double> >::iterator pointIt = closedCoordList.begin(); pointIt != closedCoordList.end(); ++pointIt) {
        annot->addCoordinate(Point(pointIt->first, pointIt->second));
      }
      annot->setGroup(grp);
      _list->addAnnotation(annot);
    }
    grp->setName(groupName + "_" + core::tostring(group_nr));
    _list->addGroup(grp);
    group_nr += 1;
	}
  return true;
}
