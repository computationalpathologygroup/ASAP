#include "UnitTest++.h"
#include "MultiResolutionImage.h"
#include "MultiResolutionImageReader.h"
#include "MultiResolutionImageWriter.h"
#include <iostream>
#include "core/filetools.h"
#include "core/PathologyEnums.h"
#include "TestData.h"
#include "config/pathology_config.h"


using namespace UnitTest;
using namespace std;
using namespace pathology;

namespace {

#ifdef WIN32
  static void reportTimeLapsed(unsigned long previous, std::string &message, unsigned long current = GetTickCount())
  {
    cout << message << " " << (current - previous) / 1000.0 << " ms" << endl;;
  }
#endif

}