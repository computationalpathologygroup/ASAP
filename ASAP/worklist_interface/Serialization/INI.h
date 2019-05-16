#ifndef __ASAP_SERIALIZATION_INI__
#define __ASAP_SERIALIZATION_INI__

#include <string>
#include <unordered_map>

namespace ASAP::Serialization::INI
{
	std::unordered_map<std::string, std::string> ParseINI(const std::string filepath);
	void WriteINI(const std::string absolute_filepath, const std::unordered_map<std::string, std::string>& records);

	namespace
	{
		void ParseLine(const std::string& line, std::unordered_map<std::string, std::string>& variable_map);
	}
}
#endif // __ASAP_SERIALIZATION_INI__