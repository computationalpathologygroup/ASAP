#include "HTTP_Connection.h"

#include <stdexcept>

#include "../Misc/StringConversions.h"

namespace ASAP::Networking
{
	HTTP_Connection::HTTP_Connection(const std::wstring base_uri, const web::http::client::http_client_config& config) : m_client_(base_uri, config)
	{
		// If the task throws an exception, assume the URL is incorrect
		try
		{
			web::http::http_request request(web::http::methods::GET);
			m_client_.request(request).wait();
		}
		catch (...)
		{
			throw std::runtime_error("Unable to connect to: " + Misc::WideStringToString(base_uri));
		}
	}

	HTTP_Connection::~HTTP_Connection(void)
	{
		CancelAllTasks();
	}

	void HTTP_Connection::CancelAllTasks(void)
	{
		m_access_mutex$.lock();
		CancelAllTasks$();
		m_access_mutex$.unlock();
	}

	void HTTP_Connection::CancelTask(const size_t task_id)
	{
		m_access_mutex$.lock();
		auto task = m_active_tasks_.find(task_id);
		if (task != m_active_tasks_.end())
		{
			if (!task->second.task.is_done())
			{
				task->second.token.cancel();
			}
			m_active_tasks_.erase(task_id);
		}
		m_access_mutex$.unlock();
	}

	size_t HTTP_Connection::QueueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer)
	{
		m_access_mutex$.lock();
		size_t token_id = m_token_counter_;
		m_active_tasks_.insert({ m_token_counter_, TokenTaskPair() });
		auto inserted_pair(m_active_tasks_.find(token_id));
		++m_token_counter_;

		// Catches the response so the attached token can be removed.
		inserted_pair->second.task = std::move(m_client_.request(request, inserted_pair->second.token.get_token()).then([observer, token_id, this](web::http::http_response response)
		{
			// Passes the response to the observer
			observer(response);

			// Remove token
			this->CancelTask(token_id);
		}));
		m_access_mutex$.unlock();

		return token_id;
	}

	pplx::task<web::http::http_response> HTTP_Connection::SendRequest(const web::http::http_request& request)
	{
		return m_client_.request(request);
	}

	void HTTP_Connection::CancelAllTasks$(void)
	{
		for (auto task : m_active_tasks_)
		{
			CancelTask(task.first);
		}
	}
}