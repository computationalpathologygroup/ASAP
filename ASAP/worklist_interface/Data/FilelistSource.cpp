#include "FilelistSource.h"

#include <functional>
#include <fstream>
#include <stdexcept>
#include <filesystem>

#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP
{
	FilelistSource::FilelistSource(const std::string filepath) : m_images(getImageFilelist(filepath))
	{
	}

	WorklistSourceInterface::SourceType FilelistSource::getSourceType(void)
	{
		return WorklistSourceInterface::SourceType::FILELIST;
	}

	size_t FilelistSource::addWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t FilelistSource::updateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t FilelistSource::deleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t FilelistSource::getWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t FilelistSource::getPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t FilelistSource::getStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t FilelistSource::getImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		receiver(m_images, 0);
		return 0;
	}

	std::set<std::string> FilelistSource::getWorklistHeaders(const DataTable::FIELD_SELECTION selectionL)
	{
		return std::set<std::string>();
	}

	std::set<std::string> FilelistSource::getPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> FilelistSource::getStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> FilelistSource::getImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	size_t FilelistSource::getImageThumbnailFile(const std::string& image_index, const std::function<void(fs::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		receiver(fs::path(*m_images.at(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	size_t FilelistSource::getImageFile(const std::string& image_index, const std::function<void(fs::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		receiver(fs::path(*m_images.at(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	DataTable FilelistSource::getImageFilelist(const std::string filepath)
	{
		std::ifstream stream(filepath);
		if (stream.is_open())
		{
			std::set<std::string> allowed_extensions = MultiResolutionImageFactory::getAllSupportedExtensions();
			DataTable images({ "id", "location", "title" });

			while (!stream.eof())
			{
				std::string line;
				std::getline(stream, line);

				fs::path image_path(line);
				if (fs::is_regular_file(image_path) && allowed_extensions.find(image_path.extension().string().substr(1)) != allowed_extensions.end())
				{
					images.insert({ std::to_string(images.size()), image_path.string(), image_path.filename().string() });
				}
			}
			return images;
		}
		throw std::runtime_error("Unable to open file: " + filepath);
	}
}