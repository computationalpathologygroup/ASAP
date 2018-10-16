#include "DirectoryDataAcquisition.h"

#include <set>
#include <boost/filesystem.hpp>

#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP::Worklist::Data
{
	DirectoryDataAcquisition::DirectoryDataAcquisition(const std::string directory_path) : m_images_(GetImageFilelist_(directory_path))
	{
	}

	WorklistDataAcquisitionInterface::SourceType DirectoryDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FILELIST;
	}

	size_t DirectoryDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetPatientRecords(const size_t worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetStudyRecords(const size_t patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetImageRecords(const size_t study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		receiver(m_images_, 0);
		return 0;
	}

	std::unordered_set<std::string> DirectoryDataAcquisition::GetWorklistHeaders(const DataTable::FIELD_SELECTION selectionL)
	{
		return std::unordered_set<std::string>();
	}

	std::unordered_set<std::string> DirectoryDataAcquisition::GetPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::unordered_set<std::string>();
	}

	std::unordered_set<std::string> DirectoryDataAcquisition::GetStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::unordered_set<std::string>();
	}

	std::unordered_set<std::string> DirectoryDataAcquisition::GetImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::unordered_set<std::string>();
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