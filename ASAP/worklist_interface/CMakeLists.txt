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
		Misc/TemporaryDirectoryTracker.h
		Misc/TemporaryDirectoryTracker.cpp
	)
	SET(GROUP_MODELS
		Models/WorklistModels.h
		Models/WorklistModels.cpp
	)
	SET(WORKLIST_RESOURCE
		Resources/WorklistInterface.qrc
	)

	IF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
		SET(GROUP_GRANDCHALLENGESUPPORT
			Data/GrandChallengeSource.h
			Data/GrandChallengeSource.cpp
			GUI/ExternalSourceDialog.h
			GUI/ExternalSourceDialog.cpp
			Networking/DjangoConnection.h
			Networking/DjangoConnection.cpp
			Networking/HTTPConnection.h
			Networking/HTTPConnection.cpp
			Networking/HTTPFileDownload.h
			Networking/HTTPFileDownload.cpp
			Serialization/JSON.h
			Serialization/JSON.cpp
		)
	ELSE(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
		SET(GROUP_GRANDCHALLENGESUPPORT "")
	ENDIF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	
	qt5_add_resources(WORKLIST_RESOURCE_ADDED ${WORKLIST_RESOURCE}) 
	
    IF(APPLE)
		set(prefix "ASAP.app/Contents")
		set(INSTALL_RUNTIME_DIR "${prefix}/MacOS")
		set(INSTALL_CMAKE_DIR "${prefix}/Resources")
		set(INSTALL_RESOURCE_DIR "${prefix}/Frameworks")
		add_executable(ASAP_browser ASAP_browser.cpp ASAP_browser.rc ${GROUP_DATA} ${GROUP_GUI} ${GROUP_MISC} ${GROUP_MODELS} ${GROUP_GRANDCHALLENGESUPPORT} ${WORKLIST_RESOURCE_ADDED} ${DIAGPathology_SOURCE_DIR}/ASAP/icon.icns)
	ELSE(APPLE)
		ADD_EXECUTABLE(ASAP_browser 
			ASAP_browser.cpp
			ASAP_browser.rc
			${GROUP_DATA}
			${GROUP_GUI}
			${GROUP_MISC}
			${GROUP_MODELS}
			${GROUP_GRANDCHALLENGESUPPORT}
			${WORKLIST_RESOURCE_ADDED}
		)
	ENDIF(APPLE)

	set_target_properties(ASAP_browser PROPERTIES FOLDER ASAP)
	set_target_properties(ASAP_browser PROPERTIES DEBUG_POSTFIX _d)

	if(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	    target_compile_definitions(ASAP_browser PRIVATE BUILD_GRANDCHALLENGE_INTERFACE)
	endif(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)

	SOURCE_GROUP("Data"				FILES ${GROUP_DATA})
	SOURCE_GROUP("GUI"				FILES ${GROUP_GUI})
	SOURCE_GROUP("Misc"				FILES ${GROUP_MISC})
	SOURCE_GROUP("Models"			FILES ${GROUP_MODELS})
	SOURCE_GROUP("GrandChallenge"   FILES ${GROUP_GRANDCHALLENGESUPPORT})

	TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE ASAPLib core multiresolutionimageinterface)
	TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::UiTools Qt5::Concurrent)
	if(NOT WIN32)
		TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE pthread)
	endif(NOT WIN32)

	IF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)
	    TARGET_LINK_LIBRARIES(ASAP_browser PRIVATE cpprestsdk::cpprest)
	ENDIF(BUILD_WORKLIST_GRANDCHALLENGE_INTERFACE)

	IF(APPLE)
		install(TARGETS ASAP_browser
				BUNDLE DESTINATION . 
				RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
				LIBRARY DESTINATION ${INSTALL_RESOURCE_DIR}
				ARCHIVE DESTINATION ${INSTALL_RESOURCE_DIR}
		)
	ELSE(APPLE)		
		install(TARGETS ASAP_browser
			EXPORT worklist-targets
			RUNTIME DESTINATION bin
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
		)
	ENDIF(APPLE)
ENDIF()