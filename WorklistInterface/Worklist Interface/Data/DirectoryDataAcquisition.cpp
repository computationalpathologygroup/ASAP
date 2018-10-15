#include "DirectoryDataAcquisition.h"

#include <functional>
#include <set>
#include <boost/filesystem.hpp>

#include "JSON_Parsing.h"
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

	std::vector<std::string> DirectoryDataAcquisition::GetPatientHeaders(void)
	{
		return std::vector<std::string>();
	}

	std::vector<std::string> DirectoryDataAcquisition::GetStudyHeaders(void)
	{
		return std::vector<std::string>();
	}

	std::vector<std::string> DirectoryDataAcquisition::GetImageHeaders(void)
	{
		return std::vector<std::string>();
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