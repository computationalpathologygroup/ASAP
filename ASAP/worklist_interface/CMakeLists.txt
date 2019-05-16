CMAKE_DEPENDENT_OPTION(BUILD_WORKLIST_INTERFACE "Builds an additional executable of ASAP that adds Worklist support." ON
						"NOT BUILD_ASAP" OFF)

if (BUILD_WORKLIST_INTERFACE)
	PROJECT(WorklistInterface)
	SET(CMAKE_AUTOMOC ON)
	SET(CMAKE_AUTOUIC ON)
	SET(Boost_USE_STATIC_LIBS OFF)
	
	FIND_PACKAGE(cpprestsdk REQUIRED)	
	FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system)

	IF (WIN32)
		SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}\ /std:c++latest)
	ELSE()
		SET(CMAKE_CXX_STANDARD 17)
	ENDIF()
	
	SET(GROUP_DATA
		worklist/Data/DataTable.h
		worklist/Data/DataTable.cpp
		worklist/Data/DirectorySource.h
		worklist/Data/DirectoryDataSource.cpp
		worklist/Data/DjangoSource.h
		worklist/Data/DjangoSource.cpp
		worklist/Data/FilelistSource.h
		worklist/Data/FilelistSource.cpp
		worklist/Data/GrandChallengeSource.h
		worklist/Data/GrandChallengeSource.cpp
		worklist/Data/SourceLoading.h
		worklist/Data/SourceLoading.cpp
		worklist/Data/WorklistSource.h
		worklist/Data/WorklistSource.cpp
	)
	SET(GROUP_GUI
		worklist/GUI/CompositeChild.h
		worklist/GUI/CompositeChild.cpp
		worklist/GUI/CompositeWindow.h
		worklist/GUI/CompositeWindow.cpp
		worklist/GUI/ExternalSourceDialog.h
		worklist/GUI/ExternalSourceDialog.cpp
		worklist/GUI/IconCreator.h
		worklist/GUI/IconCreator.cpp
		worklist/GUI/WorklistWindow.h
		worklist/GUI/WorklistWindow.cpp
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
		Serialization/Misc.h	
		Serialization/Misc.cpp
	)
	SET(RESOURCE
		resources/unavailable.png
	)
	
	ADD_EXECUTABLE(ASAP_Worklist
		main.cpp
		${GROUP_DATA}
		${GROUP_GUI}
		${GROUP_MISC}
		${GROUP_MODELS}
		${GROUP_NETWORKING}
		${GROUP_SERIALIZATION}
	)

	SOURCE_GROUP("Data"				FILES ${GROUP_DATA})
	SOURCE_GROUP("GUI"				FILES ${GROUP_GUI})
	SOURCE_GROUP("Misc"				FILES ${GROUP_MISC})
	SOURCE_GROUP("Models"			FILES ${GROUP_MODELS})
	SOURCE_GROUP("Networking"		FILES ${GROUP_NETWORKING})
	SOURCE_GROUP("Serialization"	FILES ${GROUP_SERIALIZATION})

	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC cpprestsdk::cpprest)
	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC ${Boost_LIBRARIES})	
	TARGET_LINK_LIBRARIES(ASAP_Worklist PUBLIC Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::UiTools)

	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
	
	TARGET_COMPILE_DEFINITIONS(ASAP_Worklist PUBLIC -DBOOST_ALL_DYN_LINK)
ENDIF()