#include "DocumentRetainer.h"

namespace ASAP
{
	DocumentRetainer::DocumentRetainer(void)
	{
	}

	size_t DocumentRetainer::LoadDocument(const boost::filesystem::path& filepath, const std::string& factory)
	{
		auto existing_entry = m_path_to_id_.find(filepath.string());
		if (existing_entry != m_path_to_id_.end())
		{
			return existing_entry->second;
		}

		auto result = m_documents_.insert({ m_document_counter_, Document(filepath, factory) });
		m_ptr_map_.insert({ result.first->first, std::shared_ptr<Document>(&result.first->second) });
		m_instance_counters_.insert({ result.first->first, 0 });
		m_path_to_id_.insert({ filepath.string(), result.first->first });
		++m_document_counter_;
		return result.first->first;
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
		m_instance_counters_[id] += 1;
		return DocumentInstance(m_ptr_map_[id], id, m_instance_counters_[id]);
	}

	DocumentInstance DocumentRetainer::GetDocument(const boost::filesystem::path& filepath)
	{
		size_t id = this->GetDocumentId(filepath);
		m_instance_counters_[id] += 1;
		return DocumentInstance(m_ptr_map_[id], id, m_instance_counters_[id]);
	}

	size_t DocumentRetainer::GetDocumentId(const boost::filesystem::path& filepath)
	{
		return m_path_to_id_[filepath.string()];
	}
}