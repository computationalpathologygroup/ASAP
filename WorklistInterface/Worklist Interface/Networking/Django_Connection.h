#ifndef __ASAP_NETWORKING_DJANGOCONNECTION__
#define __ASAP_NETWORKING_DJANGOCONNECTION__

#include <functional>
#include <mutex>
#include <unordered_map>

#include <cpprest/http_client.h>

#include "HTTP_Connection.h"

namespace ASAP::Networking
{
	/// <summary>
	/// Represents a connection towards a specified Django URI that handles authentication.
	/// Retains information until this object is destructed, so potential connects can be refreshed.
	/// <summary>
	class Django_Connection : public HTTP_Connection
	{
		public:
			typedef std::unordered_map<std::string, std::wstring> Credentials;
			enum AuthenticationType	{ NONE, SESSION, TOKEN };
			enum AuthenticationStatus	{ AUTHENTICATED, UNAUTHENTICATED, INVALID_CREDENTIALS };

			Django_Connection(const std::wstring base_uri, const AuthenticationType authentication_type = AuthenticationType::NONE, const Credentials credentials = Credentials(), const web::http::client::http_client_config& config = web::http::client::http_client_config());

			Credentials static CreateCredentials(const std::wstring token, const std::wstring validation_path);
			Credentials static CreateCredentials(const std::wstring username, const std::wstring password, const std::wstring csrf_path, const std::wstring auth_path);
			const Credentials& Django_Connection::SetCredentials(void);
			void SetCredentials(const Credentials credentials);

			AuthenticationStatus GetAuthenticationStatus(void) const;

			/// <summary>
			/// Allows the connection to handle the request, and returns the information to the passed observer function.
			/// </summary>
			size_t QueueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer);

			/// <summary>
			/// Sends the request through the internal client and returns the async task handling the request.
			/// </summary>
			pplx::task<web::http::http_response> SendRequest(const web::http::http_request& request);

		private:
			AuthenticationType		m_authentication_;
			Credentials				m_credentials_;
			AuthenticationStatus	m_status_;

			bool ExtractToken_(web::http::http_response& response);
			void ModifyRequest_(web::http::http_request& request);
			void SetupConnection_(void);
			void ValidateCredentials_(Credentials& credentials);
	};
}
#endif // __ASAP_NETWORKING_DJANGOCONNECTION__