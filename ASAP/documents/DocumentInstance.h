#ifndef __ASAP_DOCUMENTS_DOCUMENTINSTANCE__
#define __ASAP_DOCUMENTS_DOCUMENTINSTANCE__

#include "Document.h"

#include "asaplib_export.h"

namespace ASAP
{
	class DocumentInstance
	{
		public:
			DocumentInstance(Document& document);

			/*PluginInformation* GetPluginInformation(const std::string& plugin);
			bool HasPluginInformation(const std::string& plugin);
			void SetPluginInformation(const std::string& plugin, PluginInformation* information, const bool allow_override = false);*/

			Document&					document;
			QPointF						view_center;
			QPointF						scene_center;
			qreal						scale;


			uint64_t					current_level;
			QRect						current_fov;
			std::vector<QPainterPath>	minimap_coverage;

			//std::unordered_map<std::string, std::unique_ptr<PluginInformation>> m_plugin_information_;
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTINSTANCE__