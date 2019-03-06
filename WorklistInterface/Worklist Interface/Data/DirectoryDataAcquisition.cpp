#include "DirectoryDataAcquisition.h"

#include <set>
#include <boost/filesystem.hpp>

#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP::Data
{
	DirectoryDataAcquisition::DirectoryDataAcquisition(const std::string directory_path) : m_images_(GetImageFilelist_(directory_path))
	{
	}

	WorklistDataAcquisitionInterface::SourceType DirectoryDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FILELIST;
	}

	size_t DirectoryDataAcquisition::AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		receiver(m_images_, 0);
		return 0;
	}

	std::set<std::string> DirectoryDataAcquisition::GetWorklistHeaders(const DataTable::FIELD_SELECTION selectionL)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectoryDataAcquisition::GetPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectoryDataAcquisition::GetStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectoryDataAcquisition::GetImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	size_t DirectoryDataAcquisition::GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer)
	{
		receiver(boost::filesystem::path(*m_images_.At(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	size_t DirectoryDataAcquisition::GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer)
	{
		receiver(boost::filesystem::path(*m_images_.At(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	DataTable DirectoryDataAcquisition::GetImageFilelist_(const std::string directory_path)
	{
		std::set<std::string> allowed_extensions = MultiResolutionImageFactory::getAllSupportedExtensions();	
		DataTable images({ "id", "location", "title" });

		boost::filesystem::path directory(directory_path);
		boost::filesystem::directory_iterator end_it;
		for (boost::filesystem::directory_iterator it(directory); it != end_it; ++it)
		{
			if (boost::filesystem::is_regular_file(it->path()) && (allowed_extensions.find(it->path().extension().string().substr(1)) != allowed_extensions.end()))
			{
				images.Insert({ std::to_string(images.Size()), it->path().string(), it->path().filename().string() });
			}
		}
		return images;
	}
}