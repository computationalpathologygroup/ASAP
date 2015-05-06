#include "UnitTest++.h"
#include "TestData.h"
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
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
  return UnitTest::RunAllTests(xmlOutput.c_str(), suite.empty() ? NULL : suite.c_str());
}