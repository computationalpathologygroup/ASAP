#ifndef __ASAP_IO_HTTPCONNECTION__
#define __ASAP_IO_HTTPCONNECTION__

#include <functional>
#include <mutex>
#include <unordered_map>

#include <cpprest/http_client.h>

namespace ASAP::Worklist::IO
{
	/// <summary>
	/// Represents a connection towards a specified URI.
	/// <summary>
	class HTTP_Connection
	{
		public:
			HTTP_Connection(const std::wstring base_uri);
			~HTTP_Connection(void);

			std::wstring GetBaseURI(void);
			void SetBaseURI(const std::wstring uri);

			void CancelAllTasks(void);
			void CancelTask(const size_t task_id);

			/// <summary>
			/// Allows the connection to handle the request, and returns the information to the passed observer function.
			/// </summary>
			size_t QueueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer);

			/// <summary>
			/// Sends the request through the internal client and returns the async task handling the request.
			/// </summary>
			pplx::task<web::http::http_response> SendRequest(const web::http::http_request& request);

		private:
			struct TokenTaskPair
			{
				pplx::task<void>						task;
				concurrency::cancellation_token_source	token;
			};

			std::mutex									m_access_mutex_;
			web::http::client::http_client				m_client_;
			size_t										m_token_counter_;
			std::unordered_map<size_t, TokenTaskPair>	m_active_tasks_;

			void CancelAllTasks_(void);
	};
}
#endif // __ASAP_IO_HTTPCONNECTION__