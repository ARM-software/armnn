#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
cmake_minimum_required (VERSION 3.0.2) # 3.0.2 required for return() statement used in AddDllCopyCommands.cmake
project(pipe)

set(additional_cmake_files)
list(APPEND additional_cmake_files
    ../cmake/ArmnnVersion.cmake
    ../cmake/Utils.cmake
    ../cmake/GlobalConfig.cmake
    ../cmake/AddDllCopyCommands.cmake)

foreach(cmake_file ${additional_cmake_files})
    include(${cmake_file})
endforeach()

# Include the additional cmake files in their own target so that they will appear nicely in IDEs
add_custom_target(AdditionalCMakeFiles SOURCES ${additional_cmake_files})

include(GNUInstallDirs)

include_directories(SYSTEM common/include)
include_directories(SYSTEM client/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR})

set(BUILD_UNIT_TESTS 0)

if(BUILD_TIMELINE_DECODER)
    add_subdirectory(server/src/timelineDecoder)
    add_subdirectory(common/src)
endif()

if(BUILD_BASE_PIPE_SERVER)
    add_subdirectory(server/src/basePipeServer)
endif()
