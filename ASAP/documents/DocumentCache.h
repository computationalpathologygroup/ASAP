#ifndef __ASAP_DOCUMENTS_DOCUMENTCACHE__
#define __ASAP_DOCUMENTS_DOCUMENTCACHE__

#include <unordered_map>

#include "Document.h"
#include "DocumentInstance.h"
#include "asaplib_export.h"

#include <boost/filesystem.hpp>

namespace ASAP
{
	/// <summary>
	/// Holds and retains documents, passing them around as instances. Once all
	/// instances of a document are destroyed, the document will be unloaded.
	/// </summary>
	class ASAPLIB_EXPORT DocumentCache
	{
		public:
			/// <summary>
			/// Standard Constructor
			/// </summary>
			DocumentCache(void);

			/// <summary>
			/// Loads a document into the retainer.
			/// </summary>
			/// <param name="filepath">A filepath pointing towards an accepted ASAP image.</param>
			/// <param name="factory">The factory to utilize for the image loading.</param>
			/// <returns>The ID of the loaded document.</returns>
			size_t LoadDocument(const boost::filesystem::path& filepath, const std::string& factory = "default");
			/// <summary>
			/// Unloads or removes a document from the retainer, which will invalidate the
			/// passed id for future calls.
			/// </summary>
			/// <param name="id">The ID of the document to unload.</param>
			/// <param name="force">Whether or not to force unloading, even if document is still in use.</param>
			void UnloadDocument(const size_t id, const bool force);
			/// <summary>
			/// Acquires a document as a document instance. Throws an exception on invalid id.
			/// </summary>
			/// <param name="id">The ID of the document to acquire.</param>
			/// <returns>A document instance created from the document at id.</returns>
			DocumentInstance GetDocument(const size_t id);
			/// <summary>
			/// Acquires a document as a document instance based on the original filepath.
			/// Throws an exception on invalid path.
			/// </summary>
			/// <param name="filepath">The path of an already loaded document.</param>
			/// <returns>A document instance created from the document previously loaded.</returns>
			DocumentInstance GetDocument(const boost::filesystem::path& filepath);
			/// <summary>
			/// Acquires a document id based on the original filepath.
			/// Throws an exception on invalid path.
			/// </summary>
			/// <param name="filepath">The path of an already loaded document.</param>
			/// <returns>The id of a loaded document based on the original filepath.</returns>
			size_t GetDocumentId(const boost::filesystem::path& filepath);

		private:
			std::unordered_map<size_t, Document>					m_documents_;
			std::unordered_map <size_t, std::shared_ptr<Document>>	m_ptr_map_;
			std::unordered_map<size_t, uint16_t>					m_instance_counters_;
			std::unordered_map<std::string, size_t>					m_path_to_id_;
			size_t													m_document_counter_;
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTCACHE__