#ifndef __ASAP_NETWORKING_DJANGOCONNECTION__
#define __ASAP_NETWORKING_DJANGOCONNECTION__

#include <functional>
#include <mutex>
#include <unordered_map>

#include <cpprest/http_client.h>

#include "HTTPConnection.h"

namespace ASAP
{
	/// <summary>
	/// Represents a connection towards a specified Django URI that handles authentication.
	/// Retains information until this object is destructed, so potential connects can be refreshed.
	/// <summary>
	class DjangoConnection : public HTTPConnection
	{
		public:
			typedef std::unordered_map<std::string, std::wstring> Credentials;
			enum AuthenticationType		{ NONE, SESSION, TOKEN };

			DjangoConnection(const std::wstring base_uri, const AuthenticationType authentication_type = AuthenticationType::NONE, const Credentials credentials = Credentials(), const web::http::client::http_client_config& config = web::http::client::http_client_config());

			Credentials static CreateCredentials(const std::wstring token, const std::wstring validation_path);
			Credentials static CreateCredentials(const std::wstring username, const std::wstring password, const std::wstring csrf_path, const std::wstring auth_path);
			const Credentials& getCredentials(void);
			void setCredentials(const Credentials credentials);

			/// <summary>
			/// Allows the connection to handle the request, and returns the information to the passed observer function.
			/// </summary>
			size_t queueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer);

			/// <summary>
			/// Sends the request through the internal client and returns the async task handling the request.
			/// </summary>
			pplx::task<web::http::http_response> sendRequest(const web::http::http_request& request);

		private:
			AuthenticationType m_authentication;
			Credentials m_credentials;

			bool extractToken(web::http::http_response& response);
			void modifyRequest(web::http::http_request& request);
			void setupConnection(void);
			void validateCredentials(Credentials& credentials);
	};
}
#endif // __ASAP_NETWORKING_DJANGOCONNECTION__