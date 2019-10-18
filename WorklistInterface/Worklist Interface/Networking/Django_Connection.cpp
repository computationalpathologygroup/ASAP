#include "Django_Connection.h"

#include <functional>
#include <stdexcept>

namespace ASAP::Networking
{
	Django_Connection::Django_Connection(const std::wstring base_uri, const AuthenticationType authentication_type, const Credentials credentials, const web::http::client::http_client_config& config)
		: HTTP_Connection(base_uri, config), m_authentication_(authentication_type), m_credentials_(credentials), m_status_(UNAUTHENTICATED)
	{
		SetupConnection_();
	}

	Django_Connection::Credentials Django_Connection::CreateCredentials(const std::wstring token, const std::wstring validation_path)
	{
		Credentials credentials;
		credentials.insert({ "token", token });
		credentials.insert({ "validation", validation_path });
		return credentials;
	}

	Django_Connection::Credentials Django_Connection::CreateCredentials(const std::wstring username, const std::wstring password, const std::wstring csrf_path, const std::wstring auth_path)
	{
		Credentials credentials;
		credentials.insert({ "username", username });
		credentials.insert({ "password", password });
		credentials.insert({ "csrf", csrf_path });
		credentials.insert({ "auth", auth_path });
		return credentials;
	}

	const Django_Connection::Credentials& Django_Connection::SetCredentials(void)
	{
		return m_credentials_;
	}

	void Django_Connection::SetCredentials(const Credentials credentials)
	{
		m_access_mutex$.lock();
		CancelAllTasks$();
		m_credentials_ = credentials;
		SetupConnection_();
		m_access_mutex$.unlock();
	}

	Django_Connection::AuthenticationStatus Django_Connection::GetAuthenticationStatus(void) const
	{
		return m_status_;
	}

	size_t Django_Connection::QueueRequest(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer)
	{
		web::http::http_request authenticated_request(request);
		ModifyRequest_(authenticated_request);
		return HTTP_Connection::QueueRequest(authenticated_request, observer);
	}

	pplx::task<web::http::http_response> Django_Connection::SendRequest(const web::http::http_request& request)
	{
		web::http::http_request authenticated_request(request);
		ModifyRequest_(authenticated_request);
		return HTTP_Connection::SendRequest(authenticated_request);
	}

	bool Django_Connection::ExtractToken_(web::http::http_response& response)
	{
		auto it = response.headers().find(L"Set-Cookie");
		if (it != response.headers().end() && it->second.find(L"csrf") != std::string::npos)
		{
			m_credentials_.insert({ "token", it->second });
			return true;
		}
		return false;
	}

	void Django_Connection::ModifyRequest_(web::http::http_request& request)
	{
		if (m_authentication_ == TOKEN)
		{
			request.headers().add(L"Authorization", std::wstring(L"Token ").append(m_credentials_["token"]));
		}
		else if (m_authentication_ == SESSION)
		{
			request.headers().add(L"X-CSRFToken", m_credentials_["cookie"]);
			request.headers().add(L"CSRF_COOKIE", m_credentials_["cookie"]);
			request.headers().add(L"Cookie", L"csrf_token=" + m_credentials_["cookie"]);
		}
	}

	void Django_Connection::SetupConnection_()
	{
		// Ensures the credentials contain the required information.
		ValidateCredentials_(m_credentials_);

		try
		{
			if (m_authentication_ == TOKEN)
			{
				web::http::http_request token_test(web::http::methods::GET);
				ModifyRequest_(token_test);
				token_test.set_request_uri(m_credentials_["validation"]);

				AuthenticationStatus* status_ptr(&m_status_);
				HTTP_Connection::SendRequest(token_test).then([status_ptr](const web::http::http_response& response)
				{
					if (response.status_code() != web::http::status_codes::OK)
					{
						*status_ptr = AuthenticationStatus::INVALID_CREDENTIALS;
					}
				}).wait();
			}
			else if (m_authentication_ == SESSION)
			{
				web::http::http_request cookie_request(web::http::methods::GET);
				cookie_request.set_request_uri(m_credentials_["csrf"]);

				Credentials* cred_ptr(&m_credentials_);
				HTTP_Connection::SendRequest(cookie_request).then([cred_ptr](const web::http::http_response& response)
				{
					auto it = response.headers().find(L"Set-Cookie");
					if (it != response.headers().end() && it->second.find(L"csrf") != std::string::npos)
					{
						cred_ptr->insert({ "cookie", it->second });

						//cred_ptr->insert({ "cookie", it->second.substr(it->second.find_first_of('=') + 1, it->second.find_first_of(';') - (it->second.find_first_of('=') + 1)) });
						//cred_ptr->insert({ "cookie", L"Oxh82QwcIy7UjOVLSLysCK4Lr0DtLJKIWXICMXa8ymkCrEt00Od3lievjdXiKnrx" });
					}
				}).wait();

				if (m_credentials_.find("cookie") != m_credentials_.end())
				{
					std::wstringstream body_stream;
					body_stream << L"{ \"username\": \"" << m_credentials_["username"] << L"\", \"password\": \"" << m_credentials_["password"] << L"\" }", L"application/json";

					web::http::http_request token_request(web::http::methods::POST);
				//	token_request.set_body(body_stream.str());
					token_request.set_request_uri(m_credentials_["auth"] + L"login");


					web::http::http_response rep;
					SendRequest(token_request).then([&rep](const web::http::http_response& response)
					{
						rep = response;
					}).wait();

					std::wstring test;
					rep.extract_string().then([&test](std::wstring b)
					{
						test = b;
					}).wait();


				}
			}
		}
		catch (const std::exception& e)
		{
			m_status_ = AuthenticationStatus::INVALID_CREDENTIALS;
		}

		// Assumes all operations completed succesfully
		if (m_status_ == UNAUTHENTICATED)
		{
			m_status_ = AUTHENTICATED;
		}
	}

	void Django_Connection::ValidateCredentials_(Credentials& credentials)
	{
		std::vector<std::string> required_fields;
		if (m_authentication_ == TOKEN)
		{
			required_fields.push_back("token");
			required_fields.push_back("validation");
		}
		else if (m_authentication_ == SESSION)
		{
			required_fields.push_back("username");
			required_fields.push_back("password");
			required_fields.push_back("csrf");
			required_fields.push_back("auth");
		}

		for (const std::string& field : required_fields)
		{
			if (m_credentials_.find(field) == m_credentials_.end())
			{
				throw std::runtime_error("Incomplete authentication information.");
			}
		}
	}
}