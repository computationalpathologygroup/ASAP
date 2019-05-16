#include "INI.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace ASAP::Serialization::INI
{
	std::unordered_map<std::string, std::string> ParseINI(const std::string filepath)
	{
		std::fstream stream;
		stream.open(filepath);

		std::unordered_map<std::string, std::string> variable_map;
		if (stream.is_open())
		{
			std::string line;
			while (!stream.eof())
			{
				std::getline(stream, line);
				ParseLine(line, variable_map);
				std::fill(line.begin(), line.end(), 0);
			}
			stream.close();
		}
		else
		{
			throw std::runtime_error("Unable to open: " + filepath);
		}

		return variable_map;
	}

	void WriteINI(const std::string absolute_filepath, const std::unordered_map<std::string, std::string>& records)
	{
		std::ofstream stream;
		stream.open(absolute_filepath);

		std::unordered_map<std::string, std::string> variable_map;
		if (stream.is_open())
		{
			std::string record;
			for (auto& key_value : records)
			{
				record = key_value.first + "=" + key_value.second + '\n';
				stream.write(record.c_str(), record.size());
			}
			stream.close();
		}
		else
		{
			throw std::runtime_error("Unable to write to: " + absolute_filepath);
		}
	}

	namespace
	{
		void ParseLine(const std::string& line, std::unordered_map<std::string, std::string>& variable_map)
		{
			size_t key_start(line.find_first_not_of(' '));
			size_t value_start(line.find_first_of('=') + 1);
			size_t comment_start(line.find_first_of('#'));

			if (key_start < comment_start && value_start < comment_start)
			{
				size_t value_size = std::string::npos;
				if (comment_start != std::string::npos)
				{
					value_size = comment_start - value_start;
				}

				std::string key(line.substr(key_start, value_start - 1));
				std::string value(line.substr(value_start, value_size));
				std::transform(key.begin(), key.end(), key.begin(), ::tolower);

				if (!key.empty() && (int)key[0] > 31)
				{
					variable_map.insert({ key, value });
				}
			}
		}
	}
}