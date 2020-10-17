#include "DjangoConnection.h"

#include <functional>
#include <stdexcept>

namespace ASAP
{
	DjangoConnection::DjangoConnection(const std::wstring base_uri, const AuthenticationType authentication_type, const Credentials credentials, const web::http::client::http_client_config& config)
		: HTTPConnection(base_uri, config), m_authentication(authentication_type), m_credentials(credentials)
	{
		setupConnection();
	}

	DjangoConnection::Credentials DjangoConnection::CreateCredentials(const std::wstring token, const std::wstring validation_path)
	{
		Credentials credentials;
		credentials.insert({ "token", token });
		credentials.insert({ "validation", validation_path });
		return credentials;
	}

	DjangoConnection::Credentials DjangoConnection::CreateCredentials(const std::wstring username, const std::wstring password, const std::wstring csrf_path, const std::wstring auth_path)
	{
		Credentials credentials;
		credentials.insert({ "username", username });
		credentials.insert({ "password", password });
		credentials.insert({ "csrf", csrf_path });
		credentials.insert({ "auth", auth_path });
		return credentials;
	}

	const DjangoConnection::Credentials& DjangoConnection::getCredentials(void)
	{
		return m_credentials;
	}

	void DjangoConnection::setCredentials(const Credentials credentials)
	{
		m_access_mutex$.lock();
		cancelAllTasks$();
		m_credentials = credentials;
		setupConnection();
		m_access_mutex$.unlock();
	}

	size_t DjangoConnection::queueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer)
	{
		web::http::http_request authenticated_request(request);
		modifyRequest(authenticated_request);
		return HTTPConnection::queueRequest(authenticated_request, observer);
	}

	pplx::task<web::http::http_response> DjangoConnection::sendRequest(const web::http::http_request& request)
	{
		web::http::http_request authenticated_request(request);
		modifyRequest(authenticated_request);
		return HTTPConnection::sendRequest(authenticated_request);
	}

	bool DjangoConnection::extractToken(web::http::http_response& response)
	{
		auto it = response.headers().find(L"Set-Cookie");
		if (it != response.headers().end() && it->second.find(L"csrf") != std::string::npos)
		{
			m_credentials.insert({ "token", it->second });
			return true;
		}
		return false;
	}

	void DjangoConnection::modifyRequest(web::http::http_request& request)
	{
		if (m_authentication == TOKEN)
		{
			request.headers().add(L"Authorization", std::wstring(L"Token ").append(m_credentials["token"]));
		}
		else if (m_authentication == SESSION)
		{
			request.headers().add(L"X-CSRFToken", m_credentials["cookie"]);
			request.headers().add(L"CSRF_COOKIE", m_credentials["cookie"]);
			request.headers().add(L"Cookie", L"csrf_token=" + m_credentials["cookie"]);
		}
	}

	void DjangoConnection::setupConnection()
	{
		// Ensures the credentials contain the required information.
		validateCredentials(m_credentials);

		bool authenticated = false;
		try
		{
			if (m_authentication == TOKEN)
			{
				web::http::http_request token_test(web::http::methods::GET);
				modifyRequest(token_test);
				token_test.set_request_uri(m_credentials["validation"]);

				HTTPConnection::sendRequest(token_test).then([authenticated=&authenticated](const web::http::http_response& response)
				{
					*authenticated = response.status_code() == web::http::status_codes::OK;
				}).wait();
			}
			else if (m_authentication == SESSION)
			{
				web::http::http_request cookie_request(web::http::methods::GET);
				cookie_request.set_request_uri(m_credentials["csrf"]);

				Credentials* cred_ptr(&m_credentials);
				HTTPConnection::sendRequest(cookie_request).then([cred_ptr](const web::http::http_response& response)
				{
					auto it = response.headers().find(L"Set-Cookie");
					if (it != response.headers().end() && it->second.find(L"csrf") != std::string::npos)
					{
						cred_ptr->insert({ "cookie", it->second });
					}
				}).wait();

				if (m_credentials.find("cookie") != m_credentials.end())
				{
					// Placeholder example
					/*std::wstringstream body_stream;
					body_stream << L"{ \"username\": \"" << m_credentials_["username"] << L"\", \"password\": \"" << m_credentials_["password"] << L"\" }", L"application/json";
					web::http::http_request token_request(web::http::methods::POST);
					token_request.set_body(body_stream.str());
					token_request.set_request_uri(m_credentials_["auth"] + L"login");*/
				}
			}
		}
		catch (const std::exception& e)
		{
			// Allow it to pass, and assume invalid credentials.
		}

		if (!authenticated)
		{
			throw std::runtime_error("Unable to authenticate with source.");
		}
	}

	void DjangoConnection::validateCredentials(Credentials& credentials)
	{
		std::vector<std::string> required_fields;
		if (m_authentication == TOKEN)
		{
			required_fields.push_back("token");
			required_fields.push_back("validation");
		}
		else if (m_authentication == SESSION)
		{
			required_fields.push_back("username");
			required_fields.push_back("password");
			required_fields.push_back("csrf");
			required_fields.push_back("auth");
		}

		for (const std::string& field : required_fields)
		{
			if (m_credentials.find(field) == m_credentials.end())
			{
				throw std::runtime_error("Incomplete authentication information.");
			}
		}
	}
}