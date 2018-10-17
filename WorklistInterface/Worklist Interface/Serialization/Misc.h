#pragma once
#include <string>
#include <vector>

namespace ASAP::Worklist::Serialization::Misc
{
	std::vector<std::string> Split(const std::string& string, const char delimiter = ',', const char encapsulation = '"');
}