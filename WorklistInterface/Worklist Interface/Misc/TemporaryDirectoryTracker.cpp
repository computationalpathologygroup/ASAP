#include "TemporaryDirectoryTracker.h"

#include <map>
#include <stdexcept>

#include <boost/range/iterator_range.hpp>

namespace ASAP::Misc
{
	TemporaryDirectoryTracker::TemporaryDirectoryTracker(const boost::filesystem::path directory, const TemporaryDirectoryConfiguration configuration) : m_configuration_(configuration), m_continue_(true), m_directory_(directory)
	{
		if (boost::filesystem::exists(m_directory_) && boost::filesystem::is_regular_file(m_directory_))
		{
			throw std::runtime_error("Unable to initialize a file as temporary directory.");
		}
		else
		{
			boost::filesystem::create_directories(m_directory_);
		}

		m_update_thread_ = std::thread(&TemporaryDirectoryTracker::Update_, this);
	}

	TemporaryDirectoryTracker::~TemporaryDirectoryTracker(void)
	{
		m_continue_ = false;
		m_update_thread_.join();

		if (m_configuration_.clean_on_deconstruct)
		{
			boost::filesystem::remove_all(m_directory_);
		}
	}

	TemporaryDirectoryConfiguration TemporaryDirectoryTracker::GetStandardConfiguration(void)
	{
		return { true, true, 0, 5000 };
	}

	boost::filesystem::path TemporaryDirectoryTracker::GetAbsolutePath(void) const
	{
		return boost::filesystem::absolute(m_directory_);
	}

	std::vector<boost::filesystem::path> TemporaryDirectoryTracker::GetFilepaths(void) const
	{
		std::vector<boost::filesystem::path> filepaths;
		for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(m_directory_), { }))
		{
			filepaths.push_back(entry.path());
		}
		return filepaths;
	}

	uint64_t TemporaryDirectoryTracker::GetDirectorySizeInMb(void) const
	{
		uint64_t size = 0;
		for (boost::filesystem::recursive_directory_iterator it(m_directory_); it != boost::filesystem::recursive_directory_iterator(); ++it)
		{
			if (!boost::filesystem::is_directory(*it))
			{
				size += boost::filesystem::file_size(*it) / 1e+6;
			}
		}

		return size;
	}

	void TemporaryDirectoryTracker::Update_(void)
	{
		while (m_continue_)
		{
			size_t directory_size = GetDirectorySizeInMb();
			if (directory_size > m_configuration_.max_size_in_mb)
			{
				std::vector<boost::filesystem::path> filepaths(GetFilepaths());
				std::map<uint64_t, boost::filesystem::path*> date_sorted_files;
				for (boost::filesystem::path& p : filepaths)
				{
					date_sorted_files.insert({ static_cast<uint64_t>(boost::filesystem::last_write_time(p)), &p });
				}

				for (auto it = date_sorted_files.begin(); it != date_sorted_files.end(); ++it)
				{
					if ((directory_size <= m_configuration_.max_size_in_mb) ||
						(it == date_sorted_files.end()-- && m_configuration_.allow_overflow))
					{
						break;
					}

					directory_size -= boost::filesystem::file_size(*it->second) / 1e+6;
					boost::filesystem::remove(*it->second);
				}
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}
}