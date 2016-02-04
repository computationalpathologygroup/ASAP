#include "UnitTest++.h"
#include "XmlTestReporter.h"
#include "TestReporterStdout.h"
#include "TestData.h"
#include <iostream>
#include <fstream>
#include "config/pathology_config.h"

using namespace std;

int main(int argc, char* argv[])
{
  std::cout << "TestRunner v" << ASAP_VERSION_STRING << endl;
  if (argc < 2)
  {
	std::cout << "No data directory given, aborting" << endl;
	return 1;
  }
  string xmlOutput, suite;
  if (argc >= 3)
  {
	xmlOutput = argv[2];
	std::cout << "Writing XML to " << xmlOutput << endl;
  }
  if (argc >= 4)
  {
	suite = argv[3];
	std::cout << "Testing only suite " << suite << endl;
  }
  g_dataPath = argv[1];
  if (xmlOutput.empty()) {
    UnitTest::TestReporterStdout reporter;
    UnitTest::TestRunner runner(reporter);
    runner.RunTestsIf(UnitTest::Test::GetTestList(), suite.empty() ? NULL : suite.c_str(), UnitTest::True(), 0);
  }
  else {
    std::ofstream f(xmlOutput);
    if (f.good()) {
      UnitTest::XmlTestReporter reporter(f);
      UnitTest::TestRunner runner(reporter);
      runner.RunTestsIf(UnitTest::Test::GetTestList(), suite.empty() ? NULL : suite.c_str(), UnitTest::True(), 0);
    }
    else {
      std::cout << "Opening of XML file failed!" << endl;
    }
  }
}