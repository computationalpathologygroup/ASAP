PROJECT(WorklistInterface)

IF (WIN32)
	SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}\ /std:c++latest)
ELSE()
	SET(CMAKE_CXX_STANDARD 17)
ENDIF()

SET(CMAKE_AUTOMOC ON)


add_compile_definitions(BOOST_ALL_DYN_LINK)

SET(GROUP_DATA
	worklist/Data/DataTable.h
	worklist/Data/DataTable.cpp
	worklist/Data/DirectoryDataAcquisition.h
	worklist/Data/DirectoryDataAcquisition.cpp
	worklist/Data/DjangoDataAcquisition.h
	worklist/Data/DjangoDataAcquisition.cpp
	worklist/Data/FilelistDataAcquisition.h
	worklist/Data/FilelistDataAcquisition.cpp
	worklist/Data/SourceLoading.h
	worklist/Data/SourceLoading.cpp
	worklist/Data/Worklist_CSV_Example.h
	worklist/Data/Worklist_CSV_Example.cpp
	worklist/Data/WorklistDataAcquisitionInterface.h
	worklist/Data/WorklistDataAcquisitionInterface.cpp
)

SET(GROUP_GUI
	worklist/GUI/CompositeChild.h
	worklist/GUI/CompositeWindow.h
	worklist/GUI/ExternalSourceDialog.h
	worklist/GUI/IconCreator.h
	worklist/GUI/WorklistWindow.h
	worklist/GUI/CompositeChild.cpp
	worklist/GUI/CompositeWindow.cpp
	worklist/GUI/ExternalSourceDialog.cpp
	worklist/GUI/IconCreator.cpp
	worklist/GUI/WorklistWindow.cpp
)

SET(GROUP_SERIALIZATION
	worklist/Serialization/INI.h
	worklist/Serialization/JSON.h
	worklist/Serialization/Misc.h
	worklist/worklist/Serialization/INI.cpp
	worklist/Serialization/JSON.cpp
	worklist/Serialization/Misc.cpp
)

ADD_EXECUTABLE(ASAP
	worklist/main.cpp
	${GROUP_DATA}
	${GROUP_GUI}
	${GROUP_SERIALIZATION}
)

SOURCE_GROUP("Data"				FILES ${GROUP_DATA})
SOURCE_GROUP("GUI"				FILES ${GROUP_GUI})
SOURCE_GROUP("Serialization"	FILES ${GROUP_SERIALIZATION})