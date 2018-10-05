#include "SourceLoading.h"

#include <codecvt>
#include <boost/filesystem.hpp>

#include "DjangoDataAcquisition.h"

namespace ASAP::Worklist::Data
{
	std::unique_ptr<WorklistDataAcquisitionInterface> LoadDataSource(const std::string source_path)
	{
		boost::filesystem::path potential_system_path(source_path);
		std::unique_ptr<Data::WorklistDataAcquisitionInterface> pointer;

		if (source_path.empty())
		{
			pointer = nullptr;
		}
		else if (potential_system_path.has_extension())
		{
			// Create File Acquisition
			pointer = nullptr;
		}
		else if (source_path.find("http") != std::string::npos)
		{
			std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
			Data::DjangoRestURI uri_info = Data::DjangoDataAcquisition::GetStandardURI();
			uri_info.base_url = converter.from_bytes(source_path);

			pointer = std::unique_ptr<Data::WorklistDataAcquisitionInterface>(new Data::DjangoDataAcquisition(uri_info));
		}
		else
		{
			// Assume directory
			pointer = nullptr;
		}

		return pointer;
	}
}