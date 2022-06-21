#include "HTTPConnection.h"

#include <stdexcept>

namespace ASAP
{
	HTTPConnection::HTTPConnection(const std::wstring base_uri, const web::http::client::http_client_config& config) : m_client(base_uri, config)
	{
		// If the task throws an exception, assume the URL is incorrect
		try
		{
			web::http::http_request request(web::http::methods::GET);
			m_client.request(request).wait();
		}
		catch (...)
		{
			throw std::runtime_error("Unable to connect");
		}
	}

	HTTPConnection::~HTTPConnection(void)
	{
		cancelAllTasks();
	}

	void HTTPConnection::cancelAllTasks(void)
	{
		m_access_mutex$.lock();
		cancelAllTasks$();
		m_access_mutex$.unlock();
	}

	void HTTPConnection::cancelTask(const size_t task_id)
	{
		m_access_mutex$.lock();
		auto task = m_active_tasks.find(task_id);
		if (task != m_active_tasks.end())
		{
			if (!task->second.task.is_done())
			{
				task->second.token.cancel();
			}
			m_active_tasks.erase(task_id);
		}
		m_access_mutex$.unlock();
	}

	size_t HTTPConnection::queueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer)
	{
		m_access_mutex$.lock();
		size_t token_id = m_token_counter;
		m_active_tasks.insert({ m_token_counter, TokenTaskPair() });
		auto inserted_pair(m_active_tasks.find(token_id));
		++m_token_counter;

		// Catches the response so the attached token can be removed.
		inserted_pair->second.task = std::move(m_client.request(request, inserted_pair->second.token.get_token()).then([observer, token_id, this](web::http::http_response response)
		{
			// Passes the response to the observer
			observer(response);

			// Remove token
			this->cancelTask(token_id);
		}));
		m_access_mutex$.unlock();

		return token_id;
	}

	pplx::task<web::http::http_response> HTTPConnection::sendRequest(const web::http::http_request& request)
	{
		return m_client.request(request);
	}

	void HTTPConnection::cancelAllTasks$(void)
	{
		for (auto task : m_active_tasks)
		{
			cancelTask(task.first);
		}
	}
}