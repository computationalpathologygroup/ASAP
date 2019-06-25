#include "DocumentCollection.h"

namespace ASAP
{
	DocumentRetainer::DocumentRetainer(void)
	{
	}

	size_t DocumentRetainer::LoadDocument(const boost::filesystem::path& filepath)
	{
		auto result = m_documents_.insert({ m_id_counter_, Document(filepath) });
		m_ptr_map_.insert({ result.first->first, std::shared_ptr<Document>(&result.first->second) });
		m_path_to_id_.insert({ filepath.string(), result.first->first });
		++m_id_counter_;
	}

	void DocumentRetainer::UnloadDocument(const size_t id, const bool force)
	{
		if (m_ptr_map_[id].use_count() == 1 || force)
		{
			m_ptr_map_.erase(id);
			m_documents_.erase(id);

			for (auto it = m_path_to_id_.begin(); it != m_path_to_id_.end(); ++it)
			{
				if (it->second == id)
				{
					m_path_to_id_.erase(it);
					break;
				}
			}
		}
	}

	DocumentInstance DocumentRetainer::GetDocument(const size_t id)
	{
		return DocumentInstance(m_ptr_map_[id]);
	}

	DocumentInstance DocumentRetainer::GetDocument(const boost::filesystem::path& filepath)
	{
		return DocumentInstance(m_ptr_map_[this->GetDocumentId(filepath)]);
	}

	size_t DocumentRetainer::GetDocumentId(const boost::filesystem::path& filepath)
	{
		return m_path_to_id_[filepath.string()];
	}
}