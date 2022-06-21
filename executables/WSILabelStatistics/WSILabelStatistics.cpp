#include <string>
#include <vector>
#include <iostream>

#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "imgproc/wholeslide/LabelStatisticsWholeSlideFilter.h"
#include "core/filetools.h"
#include "core/CmdLineProgressMonitor.h"
#include "config/ASAPMacros.h"

#include "core/argparse.hpp"

using namespace std;
using namespace pathology;

int main(int argc, char *argv[]) {
  try {

      argparse::ArgumentParser desc("WSI Label Statistics", ASAP_VERSION_STRING);

      desc.add_argument("-l", "--level")
          .help("Sets pyramid level to compute on")
          .default_value(0)
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

      std::string inputPth = desc.get<std::string>("input");
      std::string outputPth = desc.get<std::string>("output");
      unsigned int level = desc.get<unsigned int>("--level");


    MultiResolutionImageReader reader; 
    std::shared_ptr<MultiResolutionImage> input = std::shared_ptr<MultiResolutionImage>(reader.open(inputPth));
    CmdLineProgressMonitor monitor;
    if (input) {
      LabelStatisticsWholeSlideFilter fltr;
      fltr.setInput(input);
      fltr.setOutput(outputPth);
      fltr.setProgressMonitor(&monitor);
      fltr.setProcessedLevel(level);
      if (!fltr.process()) {
        std::cerr << "ERROR: Processing failed" << std::endl;
      }
    }
    else {
      std::cerr << "ERROR: Invalid input image" << std::endl;
    }
  } 
  catch (std::exception& e) {
    std::cerr << "Unhandled exception: "
      << e.what() << ", application will now exit" << std::endl;
    return 2;
  }
	return 0;
}



