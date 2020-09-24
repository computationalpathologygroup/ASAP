INCLUDE(CMakeDependentOption)

CMAKE_DEPENDENT_OPTION(BUILD_WORKLIST_INTERFACE "Builds an additional executable of ASAP that adds Worklist support." OFF
						"BUILD_ASAP" OFF)
CMAKE_DEPENDENT_OPTION(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE "Adds Grand Challenge support to ASAP." OFF
						"BUILD_WORKLIST_INTERFACE" OFF)

if (BUILD_WORKLIST_INTERFACE)
	SET(CMAKE_VERBOSE_MAKEFILE ON)
	SET(CMAKE_AUTOMOC ON)
	SET(CMAKE_AUTOUIC ON)
	SET(CMAKE_AUTORCC ON)

	PROJECT(WorklistInterface)

	IF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	FIND_PACKAGE(cpprestsdk REQUIRED)
	ENDIF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)

	SET(GROUP_DATA
		Data/DataTable.h
		Data/DataTable.cpp
		Data/DirectorySource.h
		Data/DirectorySource.cpp
		Data/FilelistSource.h
		Data/FilelistSource.cpp
		Data/SourceProxy.h
		Data/SourceProxy.cpp
		Data/WorklistSourceInterface.h
		Data/WorklistSourceInterface.cpp
	)
	SET(GROUP_GUI
		GUI/CompositeChild.h
		GUI/CompositeChild.cpp
		GUI/CompositeWindow.h
		GUI/CompositeWindow.cpp
		GUI/IconCreator.h
		GUI/IconCreator.cpp
		GUI/WorklistWindow.h
		GUI/WorklistWindow.cpp
		GUI/ThumbnailCache.h
		GUI/ThumbnailCache.cpp
		GUI/CompositeWindowLayout.ui
		GUI/WorklistWindowLayout.ui
	)
	SET(GROUP_MISC
		Misc/StringConversions.h
		Misc/StringConversions.cpp
		Misc/StringManipulation.h
		Misc/StringManipulation.cpp
		Misc/TemporaryDirectoryTracker.h
		Misc/TemporaryDirectoryTracker.cpp
	)
	SET(GROUP_MODELS
		Models/WorklistModels.h
		Models/WorklistModels.cpp
	)
	SET(GROUP_SERIALIZATION
		Serialization/INI.h
		Serialization/INI.cpp
	)
	SET(WORKLIST_RESOURCE
		resources/WorklistInterface.qrc
	)

	IF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
		SET(GROUP_GRANDCHALLENGESUPPORT
			Data/GrandChallengeSource.h
			Data/GrandChallengeSource.cpp
			GUI/ExternalSourceDialog.h
			GUI/ExternalSourceDialog.cpp
			Networking/Django_Connection.h
			Networking/Django_Connection.cpp
			Networking/HTTP_Connection.h
			Networking/HTTP_Connection.cpp
			Networking/HTTP_File_Download.h
			Networking/HTTP_File_Download.cpp
			Serialization/JSON.h
			Serialization/JSON.cpp
		)
		SET(GROUP_GRANDCHALLENGESUPPORT "")
	ELSE(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	ENDIF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	
	qt5_add_resources(WORKLIST_RESOURCE_ADDED ${WORKLIST_RESOURCE}) 
	
	ADD_EXECUTABLE(ASAP_browser 
		ASAP_browser.cpp
		ASAP_browser.rc
		${GROUP_DATA}
		${GROUP_GUI}
		${GROUP_MISC}
		${GROUP_MODELS}
		${GROUP_SERIALIZATION}
		${GROUP_GRANDCHALLENGESUPPORT}
		${WORKLIST_RESOURCE_ADDED}
	)
	set_target_properties(ASAP_browser PROPERTIES FOLDER ASAP)
	set_target_properties(ASAP_browser PROPERTIES DEBUG_POSTFIX _d)

	if(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	    target_compile_definitions(ASAP_browser PRIVATE BUILD_GRANDCHALLENGE_INTERFACE)
	endif(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)

	SOURCE_GROUP("Data"				FILES ${GROUP_DATA})
	SOURCE_GROUP("GUI"				FILES ${GROUP_GUI})
	SOURCE_GROUP("Misc"				FILES ${GROUP_MISC})
	SOURCE_GROUP("Models"			FILES ${GROUP_MODELS})
	SOURCE_GROUP("Serialization"	FILES ${GROUP_SERIALIZATION})
	SOURCE_GROUP("GrandChallenge"   FILES ${GROUP_GRANDCHALLENGESUPPORT})

	TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE ASAPLib core multiresolutionimageinterface)
	TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::UiTools Qt5::Concurrent)
	TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE Boost::disable_autolinking Boost::filesystem)	

	IF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	    TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE cpprestsdk::cpprest)
	ENDIF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
		
	install(TARGETS ASAP_browser
        EXPORT worklist-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
ENDIF()