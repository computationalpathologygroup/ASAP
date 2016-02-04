#include <iostream>
#include <string>
#include <vector>

#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "OpenSlideImage.h"
#include "MultiResolutionImageWriter.h"
#include "AperioSVSWriter.h"
#include "core/filetools.h"
#include "core/PathologyEnums.h"
#include "core/CmdLineProgressMonitor.h"
#include <sstream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace pathology;

void convertImage(std::string fileIn, std::string fileOut, bool svs = false, std::string compression = "LZW", double quality = 70., double spacingX = -1.0, double spacingY = -1.0) {
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
        writer->setTileSize(512);
        if (compression == string("LZW")) {
          writer->setCompression(LZW);
        } 
        else if (compression == string("RAW")) {
          writer->setCompression(RAW);
        }
        else if (compression == string("JPEG")) {
          writer->setCompression(JPEG);
        }
        else if (compression == string("JPEG2000Lossless")) {
          writer->setCompression(JPEG2000_LOSSLESS);
        }
        else if (compression == string("JPEG2000Lossy")) {
          writer->setCompression(JPEG2000_LOSSY);
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

        if (spacingX > 0.0 && spacingY > 0.0) {
          std::vector<double> overrideSpacing;
          overrideSpacing.push_back(spacingX);
          overrideSpacing.push_back(spacingY);
          writer->setOverrideSpacing(overrideSpacing);
        }
        CmdLineProgressMonitor* monitor = new CmdLineProgressMonitor();
        monitor->setStatus("Processing " + fileIn);
        writer->setProgressMonitor(monitor);
        if (dynamic_cast<OpenSlideImage*>(img)) {
          dynamic_cast<OpenSlideImage*>(img)->setIgnoreAlpha(true);
        }
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

    std::string inputPth, outputPth, codec;
    double rate, spacingX, spacingY;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Displays this message")
      ("svs,s", "Convert to Aperio SVS instead of regular TIFF")
      ("codec,c", po::value<std::string>(&codec)->default_value("LZW"), "Set compression codec. Can be one of the following: RAW, LZW, JPEG, JPEG2000Lossless or JPEG2000Lossy")
      ("rate,r", po::value<double>(&rate)->default_value(70.), "Set compression rate for JPEG and JPEG2000Lossy")
      ("spacingX,x", po::value<double>(&spacingX)->default_value(-1.0), "Set the pixel spacing of the x-dimension")
      ("spacingY,y", po::value<double>(&spacingY)->default_value(-1.0), "Set the pixel spacing of the y-dimension")
      ;
  
    po::positional_options_description positionalOptions;
    positionalOptions.add("input", 1);
    positionalOptions.add("output", 1);

    po::options_description posDesc("Positional descriptions");
    posDesc.add_options()
      ("input", po::value<std::string>(&inputPth)->required(), "Path to input")
      ("output", po::value<std::string>(&outputPth)->default_value("."), "Path to output")
      ;


    po::options_description descAndPos("All options");
    descAndPos.add(desc).add(posDesc);

    po::variables_map vm;
    try {
      po::store(po::command_line_parser(argc, argv).options(descAndPos)
        .positional(positionalOptions).run(),
        vm);
      if (!vm.count("input")) {
        cout << "MultiResolutionImageConverter v" << ASAP_VERSION_STRING << endl;
        cout << "Usage: MultiResImageConverter.exe input output [options]" << endl;
      }
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
      }
      po::notify(vm);
    }
    catch (boost::program_options::required_option& e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << "Use -h or --help for usage information" << std::endl;
      return 1;
    }

    bool svs = false;
    if (vm.count("svs")) {
      svs = true;
    }

    if (core::fileExists(inputPth) && !core::dirExists(outputPth)) {
      if (!vm["spacingX"].defaulted() || !vm["spacingY"].defaulted()) {
        convertImage(inputPth, outputPth, svs, codec, rate, spacingX, spacingY);
      }
      else {
        convertImage(inputPth, outputPth, svs, codec, rate);
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
        if (!vm["spacingX"].defaulted() || !vm["spacingY"].defaulted()) {
          convertImage(fls[i], outPth, svs, codec, rate, spacingX, spacingY);
        }
        else {
          convertImage(fls[i], outPth, svs, codec, rate);
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



