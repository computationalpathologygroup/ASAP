#ifndef __ASAP_NETWORKING_HTTPCONNECTION__
#define __ASAP_NETWORKING_HTTPCONNECTION__

#include <functional>
#include <mutex>
#include <unordered_map>

#include <cpprest/http_client.h>

namespace ASAP
{
	/// <summary>
	/// Represents a connection towards a specified URI.
	/// <summary>
	class HTTPConnection
	{
		public:
			HTTPConnection(const std::wstring base_uri, const web::http::client::http_client_config& config = web::http::client::http_client_config());
			~HTTPConnection(void);

			void cancelAllTasks(void);
			void cancelTask(const size_t task_id);

			/// <summary>
			/// Allows the connection to handle the request, and returns the information to the passed observer function.
			/// </summary>
			size_t queueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer);

			/// <summary>
			/// Sends the request through the internal client and returns the async task handling the request.
			/// </summary>
			pplx::task<web::http::http_response> sendRequest(const web::http::http_request& request);
		
		protected:
			std::mutex m_access_mutex$;

			void cancelAllTasks$(void);

		private:
			struct TokenTaskPair
			{
				pplx::task<void> task;
				concurrency::cancellation_token_source	token;
			};

			web::http::client::http_client m_client;
			size_t m_token_counter;
			std::unordered_map<size_t, TokenTaskPair>	m_active_tasks;
	};
}
#endif // __ASAP_NETWORKING_HTTPCONNECTION__