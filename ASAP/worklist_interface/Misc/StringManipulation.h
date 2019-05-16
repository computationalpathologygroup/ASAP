#ifndef __ASAP_MISC_STRINGMANIPULATION__
#define __ASAP_MISC_STRINGMANIPULATION__

#include <string>
#include <vector>

namespace ASAP::Misc
{
	std::vector<std::string> Split(const std::string& string, const char delimiter = ',', const char encapsulation = '"');
}
#endif // __ASAP_SERIALIZATION_JSON__