#ifndef __ASAP_DOCUMENTS_DOCUMENTINSTANCE__
#define __ASAP_DOCUMENTS_DOCUMENTINSTANCE__

#include <unordered_map>

#include "Document.h"
#include "../PathologyViewState.h"
#include "../PluginState.h"

#include "asaplib_export.h"

namespace ASAP
{
	class DocumentInstance
	{
		public:
			DocumentInstance(Document& document, const size_t document_id = 0, const uint16_t instance_id = 1);
			DocumentInstance(std::shared_ptr<Document> document, const size_t document_id = 0, const uint16_t instance_id = 1);
			DocumentInstance(std::weak_ptr<Document> document, const size_t document_id = 0, const uint16_t instance_id = 1);

			const PluginInformation* GetPluginInformation(const std::string& plugin) const;
			bool HasPluginInformation(const std::string& plugin) const;
			void SetPluginInformation(const std::string& plugin, PluginInformation* information, const bool allow_override = false);
			
			std::shared_ptr<Document>	document;
			const std::string			document_id;
			const std::string			name;
			PathologyViewState			view_state;


			// TODO: Remove these
			uint64_t					current_level;
			QRect						current_fov;
			std::vector<QPainterPath>	minimap_coverage;

			
		private:
			std::unordered_map<std::string, PluginState> m_plugin_information_;

			static std::string GetInstanceName_(Document& document, const uint16_t instance_id);
			void SetupInstance_(void);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTINSTANCE__