INCLUDE(CMakeDependentOption)

CMAKE_DEPENDENT_OPTION(BUILD_WORKLIST_INTERFACE "Builds an additional executable of ASAP that adds Worklist support." OFF
						"BUILD_ASAP" OFF)

if (BUILD_WORKLIST_INTERFACE)
	SET(CMAKE_VERBOSE_MAKEFILE ON)
	SET(CMAKE_AUTOMOC ON)
	SET(CMAKE_AUTOUIC ON)
	SET(CMAKE_AUTORCC ON)

	PROJECT(WorklistInterface)

	FIND_PACKAGE(cpprestsdk REQUIRED)	

	SET(GROUP_DATA
		Data/DataTable.h
		Data/DataTable.cpp
		Data/DirectorySource.h
		Data/DirectorySource.cpp
		Data/FilelistSource.h
		Data/FilelistSource.cpp
		Data/GrandChallengeSource.h
		Data/GrandChallengeSource.cpp
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
		GUI/ExternalSourceDialog.h
		GUI/ExternalSourceDialog.cpp
		GUI/IconCreator.h
		GUI/IconCreator.cpp
		GUI/WorklistWindow.h
		GUI/WorklistWindow.cpp
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
	SET(GROUP_NETWORKING
		Networking/Django_Connection.h
		Networking/Django_Connection.cpp
		Networking/HTTP_Connection.h
		Networking/HTTP_Connection.cpp
		Networking/HTTP_File_Download.h
		Networking/HTTP_File_Download.cpp
	)
	SET(GROUP_SERIALIZATION
		Serialization/INI.h
		Serialization/INI.cpp
		Serialization/JSON.h
		Serialization/JSON.cpp
	)
	SET(RESOURCE
		resources/WorklistInterface.qrc
	)
	
	qt5_add_resources(RESOURCE_ADDED ${RESOURCE}) 
	
	ADD_EXECUTABLE(ASAP_Worklist WIN32
		main.cpp
		${GROUP_DATA}
		${GROUP_GUI}
		${GROUP_MISC}
		${GROUP_MODELS}
		${GROUP_NETWORKING}
		${GROUP_SERIALIZATION}
		${RESOURCE_ADDED}
	)

	SOURCE_GROUP("Data"				FILES ${GROUP_DATA})
	SOURCE_GROUP("GUI"				FILES ${GROUP_GUI})
	SOURCE_GROUP("Misc"				FILES ${GROUP_MISC})
	SOURCE_GROUP("Models"			FILES ${GROUP_MODELS})
	SOURCE_GROUP("Networking"		FILES ${GROUP_NETWORKING})
	SOURCE_GROUP("Serialization"	FILES ${GROUP_SERIALIZATION})

	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC ASAPLib core multiresolutionimageinterface)
	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC cpprestsdk::cpprest)
	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC ${Boost_LIBRARIES})	
	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::UiTools)
	
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
	
	add_definitions(-DBOOST_ALL_DYN_LINK)
	
	install(TARGETS ASAP_Worklist
        EXPORT worklist-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
ENDIF()