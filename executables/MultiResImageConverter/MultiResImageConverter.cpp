#include <iostream>
#include <string>
#include <vector>

#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/OpenSlideImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/AperioSVSWriter.h"
#include "core/filetools.h"
#include "core/PathologyEnums.h"
#include "core/CmdLineProgressMonitor.h"
#include "config/ASAPMacros.h"
#include <sstream>

#include "core/argparse.hpp"

using namespace std;
using namespace pathology;

void convertImage(std::string fileIn, std::string fileOut, bool svs = false, std::string compression = "LZW", double quality = 70., double spacingX = -1.0, double spacingY = -1.0, unsigned int tileSize = 512, int maxPyramidLevels = -1, int downsamplePerLevel =2) {
  MultiResolutionImageReader read;
  MultiResolutionImageWriter* writer;
  if (svs) {
    writer = new AperioSVSWriter();
  }
  else {
    writer = new MultiResolutionImageWriter();
  }
  if (core::fileExists(fileIn)) {
    MultiResolutionImage* img = read.open(fileIn);
    if (img) {
      if (img->valid()) {
        writer->setTileSize(tileSize);
        if (compression == string("LZW")) {
          writer->setCompression(LZW);
        } 
        else if (compression == string("RAW")) {
          writer->setCompression(RAW);
        }
        else if (compression == string("JPEG")) {
          writer->setCompression(JPEG);
        }
        else if (compression == string("JPEG2000")) {
          writer->setCompression(JPEG2000);
        }
        else {
          cout << "Invalid compression, setting default LZW as compression" << endl;
          writer->setCompression(LZW);
        }
        if (quality > 100) {
          cout << "Too high rate, maximum is 100, setting to 100 (for JPEG2000 this is equal to lossless)" << endl; 
          writer->setJPEGQuality(100);
        } else if (quality <= 0.001) {
          cout << "Too low rate, minimum is 0.001, setting to 1" << endl;
          writer->setJPEGQuality(1);
        } else {
          writer->setJPEGQuality(quality);
        }

        writer->setDownsamplePerLevel(downsamplePerLevel);
        writer->setMaxNumberOfPyramidLevels(maxPyramidLevels);

        if (spacingX > 0.0 && spacingY > 0.0) {
          std::vector<double> overrideSpacing;
          overrideSpacing.push_back(spacingX);
          overrideSpacing.push_back(spacingY);
          writer->setOverrideSpacing(overrideSpacing);
        }
        CmdLineProgressMonitor* monitor = new CmdLineProgressMonitor();
        monitor->setStatus("Processing " + fileIn);
        writer->setProgressMonitor(monitor);
        writer->writeImageToFile(img, fileOut);
        delete monitor;
      }
      else {
        cout << "Input file not valid" << endl;
      }
    }
    else {
      cout << "Input file not compatible" << endl;
    }
  }
  else {
    cout << "Input file does not exist" << endl;
  }
}

int main(int argc, char *argv[]) {
  try {

    argparse::ArgumentParser desc("Multi-resolution image converter");

    desc.add_argument("-s", "--svs")
        .help("Convert to Aperio SVS instead of regular TIFF")
        .default_value(false)
        .implicit_value(true);

    desc.add_argument("-c", "--codec")
        .help("Set compression codec. Can be one of the following: RAW, LZW, JPEG, JPEG2000")
        .default_value("LZW");

    desc.add_argument("-r", "--rate")
        .help("Set compression rate for JPEG and JPEG2000")
        .default_value(70.)
        .scan<'g', double>();

    desc.add_argument("-x", "--spacingX")
        .help("Set the pixel spacing of the x-dimension")
        .default_value(-1)
        .scan<'g', double>();

    desc.add_argument("-y", "--spacingY")
        .help("Set the pixel spacing of the y-dimension")
        .default_value(-1)
        .scan<'g', double>();

    desc.add_argument("-t", "--tileSize")
        .help("Sets the tile size for the TIFF")
        .default_value(512)
        .scan<'i', unsigned int>();

    desc.add_argument("-p", "--pyramidLevels")
        .help("Sets the maximum number of pyramid levels; -1 indicates that the number of levels is automatically determined")
        .default_value(-1)
        .scan<'i', int>();

    desc.add_argument("-d", "--downsample")
        .help("Sets the downsample factor between each pyramid level")
        .default_value(2)
        .scan<'i', unsigned int>();

    desc.add_argument("input")
        .help("Path to the input image")
        .required();

    desc.add_argument("output")
        .help("Path to the output image")
        .default_value(".");

    try {
        desc.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << desc;
        std::exit(1);
    }

    bool svs = false;
    if (desc["--svs"] == true) {
      svs = true;
    }

    std::string inputPth = desc.get<std::string>("input");
    std::string outputPth = desc.get<std::string>("output");
    double spacingX = desc.get<double>("--spacingX");
    double spacingY = desc.get<double>("--spacingY");
    double rate = desc.get<double>("--rate");
    std::string codec = desc.get<std::string>("--codec");
    unsigned int tileSize = desc.get<unsigned int>("--tileSize");
    unsigned int downsamplePerLevel = desc.get<unsigned int>("--downsample");
    int pyramidLevels = desc.get<int>("--pyramidLevels");

    if (core::fileExists(inputPth) && !core::dirExists(outputPth)) {
      if (desc.is_used("--spacingX") || desc.is_used("--spacingY")) {
        convertImage(inputPth, outputPth, svs, codec, rate, spacingX, spacingY, tileSize, pyramidLevels, downsamplePerLevel);
      }
      else {
        convertImage(inputPth, outputPth, svs, codec, rate, -1., -1., tileSize, pyramidLevels, downsamplePerLevel);
      }
    } 
    else if (core::dirExists(outputPth)) { //Could be wildcards and output dir 
      std::string pth = core::extractFilePath(inputPth);
      std::string query = core::extractFileName(inputPth);
      vector<string> fls;
      core::getFiles(pth, query, fls);
      for (int i = 0; i < fls.size(); ++i) {
        string outPth = fls[i];
        core::changePath(outPth, outputPth);
        if (svs) {
          core::changeExtension(outPth, "svs");
        }
        else {
          core::changeExtension(outPth, "tif");
        }
        if (desc.is_used("--spacingX") || desc.is_used("--spacingY")) {
          convertImage(fls[i], outPth, svs, codec, rate, spacingX, spacingY, tileSize, pyramidLevels, downsamplePerLevel);
        }
        else {
          convertImage(fls[i], outPth, svs, codec, rate, -1., -1., tileSize, pyramidLevels, downsamplePerLevel);
        }
      }
    }
  } 
  catch (std::exception& e) {
    std::cerr << "Unhandled exception: "
      << e.what() << ", application will now exit" << std::endl;
    return 2;
  }
	return 0;
}



