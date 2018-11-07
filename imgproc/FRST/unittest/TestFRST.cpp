#include "UnitTest++.h"
#include "FRST.h"
#include "TestData.h"
#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#include <shellapi.h>
#endif
#include <iostream>

using namespace UnitTest;
using namespace std;
using namespace cv;

namespace {

#ifdef WIN32
  static void reportTimeLapsed(unsigned long previous, std::string &message, unsigned long current = GetTickCount())
  {
    cout << message << " " << static_cast<float>(current - previous) / 1000.0 << " s" << endl;
  }
#endif

}