#ifndef __ASAP_MISC_TEMPORARYDIRECTORYTRACKER__
#define __ASAP_MISC_TEMPORARYDIRECTORYTRACKER__

#include <thread>
#include <vector>

#include <boost/filesystem.hpp>

namespace ASAP
{
	// TODO: Perhaps expand this class to handle a set of directoryies

	/// <summary>
	/// Holds variables that modify the behaviour of the temporary directory tracker.
	/// Setting any of the integer variables to 0 will effectively mean infinite space or files.
	///
	/// Warning: allow_overflow allows for a single file to be bigger than the directory,
	/// setting this to false will remove a file if it exceeds the maximum size.
	/// </summary>
	struct TemporaryDirectoryConfiguration
	{
		bool		allow_overflow;
		bool		clean_on_deconstruct;
		uint32_t	max_files;
		uint64_t	max_size_in_mb;
	};

	class TemporaryDirectoryTracker
	{
		public:
			TemporaryDirectoryTracker(const boost::filesystem::path directory, const TemporaryDirectoryConfiguration configuration = getStandardConfiguration());
			~TemporaryDirectoryTracker(void);

			static TemporaryDirectoryConfiguration getStandardConfiguration(void);

			boost::filesystem::path getAbsolutePath(void) const;
			std::vector<boost::filesystem::path> getFilepaths(void) const;
			uint64_t getDirectorySizeInMb(void) const;

		private:
			TemporaryDirectoryConfiguration m_configuration;
			bool							m_continue;
			boost::filesystem::path			m_directory;
			std::thread						m_update_thread;

			void update(void);
	};
}
#endif // __ASAP_MISC_TEMPORARYDIRECTORYTRACKER__