#include "FilelistDataAcquisition.h"

#include <functional>
#include <fstream>
#include <stdexcept>
#include <boost/filesystem.hpp>

#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP::Worklist::Data
{
	FilelistDataAcquisition::FilelistDataAcquisition(const std::string filepath) : m_images_(GetImageFilelist_(filepath))
	{
	}

	WorklistDataAcquisitionInterface::SourceType FilelistDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FILELIST;
	}

	size_t FilelistDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t FilelistDataAcquisition::GetPatientRecords(const size_t worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t FilelistDataAcquisition::GetStudyRecords(const size_t patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t FilelistDataAcquisition::GetImageRecords(const size_t study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		receiver(m_images_, 0);
		return 0;
	}

	std::set<std::string> FilelistDataAcquisition::GetWorklistHeaders(const DataTable::FIELD_SELECTION selectionL)
	{
		return std::set<std::string>();
	}

	std::set<std::string> FilelistDataAcquisition::GetPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> FilelistDataAcquisition::GetStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> FilelistDataAcquisition::GetImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	DataTable FilelistDataAcquisition::GetImageFilelist_(const std::string filepath)
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

				boost::filesystem::path image_path(line);
				if (boost::filesystem::is_regular_file(image_path) && allowed_extensions.find(image_path.extension().string().substr(1)) != allowed_extensions.end())
				{
					images.Insert({ std::to_string(images.Size()), image_path.string(), image_path.filename().string() });
				}
			}
			return images;
		}
		throw std::runtime_error("Unable to open file: " + filepath);
	}
}