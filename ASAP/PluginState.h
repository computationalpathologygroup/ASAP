#ifndef __PLUGIN_STATE__
#define __PLUGIN_STATE__

#include <memory>

#include "asaplib_export.h"

namespace ASAP
{
	/// <summary>
	/// Represents the base plugin information, it's meant to be inherited
	/// and expanded upon to carry the required information.
	/// <summary>
	class ASAPLIB_EXPORT PluginInformation
	{
	};

	/// <summary>
	/// Holds the plugin information as a state, offering move or copy operations
	/// while retaining the ability to deal with a unique pointer holding the
	/// PluginInformation pointer.
	/// <summary>
	class ASAPLIB_EXPORT PluginState
	{
		public:
			std::unique_ptr<PluginInformation> information;

			/// <summary>
			/// Default constructor, inserts the information pointer into an unique pointer.
			/// </summary>
			/// <param name="information">A pointer towards the object holding the plugin information.</param>
			PluginState(PluginInformation* information);
			/// <summary>
			/// Copy constructor, initializes another instance of the PluginInformation.
			/// </summary>
			/// <param name="other">Another PluginState.</param>
			PluginState(const PluginState& other);
			/// <summary>
			/// Move constructor, moves the existing information into the new object.
			/// </summary>
			/// <param name="other">Another PluginState that is moved.<param>
			PluginState(PluginState&& other);
			/// <summary>
			/// The default destructor.
			/// </summary>
			~PluginState(void);

			/// <summary>
			/// Copy operator, initializes another instance of the PluginInformation.
			/// </summary>
			/// <param name="other">Another PluginState.<param>
			PluginState& operator=(const PluginState& other);
			/// <summary>
			/// Move operator, moves the existing information into the new object.
			/// </summary>
			/// <param name="other">Another PluginState that is moved.<param>
			PluginState& operator=(PluginState&& other);
	};
}
#endif // __PLUGIN_STATE__