#include "TemporaryDirectoryTracker.h"

#include <map>
#include <stdexcept>

#include <boost/range/iterator_range.hpp>

namespace ASAP
{
	TemporaryDirectoryTracker::TemporaryDirectoryTracker(const boost::filesystem::path directory, const TemporaryDirectoryConfiguration configuration) : m_configuration(configuration), m_continue(true), m_directory(directory)
	{
		if (boost::filesystem::exists(m_directory) && boost::filesystem::is_regular_file(m_directory))
		{
			throw std::runtime_error("Unable to initialize a file as temporary directory.");
		}
		else
		{
			boost::filesystem::create_directories(m_directory);
		}

		m_update_thread = std::thread(&TemporaryDirectoryTracker::update, this);
	}

	TemporaryDirectoryTracker::~TemporaryDirectoryTracker(void)
	{
		m_continue = false;
		m_update_thread.join();

		if (m_configuration.clean_on_deconstruct)
		{
			boost::filesystem::remove_all(m_directory);
		}
	}

	TemporaryDirectoryConfiguration TemporaryDirectoryTracker::getStandardConfiguration(void)
	{
		return { true, true, 0, 5000 };
	}

	boost::filesystem::path TemporaryDirectoryTracker::getAbsolutePath(void) const
	{
		return boost::filesystem::absolute(m_directory);
	}

	std::vector<boost::filesystem::path> TemporaryDirectoryTracker::getFilepaths(void) const
	{
		std::vector<boost::filesystem::path> filepaths;
		for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(m_directory), { }))
		{
			filepaths.push_back(entry.path());
		}
		return filepaths;
	}

	uint64_t TemporaryDirectoryTracker::getDirectorySizeInMb(void) const
	{
		uint64_t size = 0;
		for (boost::filesystem::recursive_directory_iterator it(m_directory); it != boost::filesystem::recursive_directory_iterator(); ++it)
		{
			if (!boost::filesystem::is_directory(*it))
			{
				size += boost::filesystem::file_size(*it) / 1e+6;
			}
		}

		return size;
	}

	void TemporaryDirectoryTracker::update(void)
	{
		while (m_continue)
		{
			size_t directory_size = getDirectorySizeInMb();
			if (directory_size > m_configuration.max_size_in_mb)
			{
				std::vector<boost::filesystem::path> filepaths(getFilepaths());
				std::map<uint64_t, boost::filesystem::path*> date_sorted_files;
				for (boost::filesystem::path& p : filepaths)
				{
					date_sorted_files.insert({ static_cast<uint64_t>(boost::filesystem::last_write_time(p)), &p });
				}

				for (auto it = date_sorted_files.begin(); it != date_sorted_files.end(); ++it)
				{
					if ((directory_size <= m_configuration.max_size_in_mb) ||
						(it == date_sorted_files.end()-- && m_configuration.allow_overflow))
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