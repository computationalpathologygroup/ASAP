#include "StringManipulation.h"

namespace ASAP::Misc
{
	std::vector<std::string> Split(const std::string& string, const char delimiter, const char encapsulation)
	{
		std::vector<std::string> split_string;

		size_t value_start	= 0;
		size_t value_end	= 0;
		bool within_encapsulation = false;
		for (size_t current_char = 0; current_char < string.size(); ++current_char)
		{
			if (string[current_char] == encapsulation)
			{
				within_encapsulation = !within_encapsulation;
				if (within_encapsulation)
				{
					value_start = current_char + 1;
				}
				else
				{
					value_end = current_char;
				}
			}
			else if (string[current_char] == delimiter && value_end == 0)
			{
				value_end = current_char;
			}

			if (value_start < value_end)
			{
				split_string.push_back(string.substr(value_start, value_end - value_start));
				value_start = current_char + 1;
				value_end	= 0;
			}
		}

		if (value_start < string.size() && (string[string.size() - 1] != delimiter || string[string.size() - 1] != encapsulation))
		{
			value_end = string.size();
			split_string.push_back(string.substr(value_start, value_end - value_start));
		}

		return split_string;
	}
}