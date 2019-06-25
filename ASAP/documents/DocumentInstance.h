#ifndef __ASAP_DOCUMENTS_DOCUMENTINSTANCE__
#define __ASAP_DOCUMENTS_DOCUMENTINSTANCE__

#include <memory>
#include "Document.h"
#include "../PathologyViewState.h"

#include "asaplib_export.h"

namespace ASAP
{
	class DocumentInstance
	{
		public:
			DocumentInstance(Document& document);
			DocumentInstance(std::shared_ptr<Document> document);
			DocumentInstance(std::weak_ptr<Document> document);

			/*PluginInformation* GetPluginInformation(const std::string& plugin);
			bool HasPluginInformation(const std::string& plugin);
			void SetPluginInformation(const std::string& plugin, PluginInformation* information, const bool allow_override = false);*/

			const std::shared_ptr<Document>	document;
			PathologyViewState				view_state;

			uint64_t					current_level;
			QRect						current_fov;
			std::vector<QPainterPath>	minimap_coverage;

			//std::unordered_map<std::string, std::unique_ptr<PluginInformation>> m_plugin_information_;
		private:
			void SetupInstance_(void);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTINSTANCE__