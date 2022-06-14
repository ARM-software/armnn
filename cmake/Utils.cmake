#
# Copyright © 2018 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Function which creates appropriate "source groups" (filter folders in Visual Studio) for the given list of source files
function(createSourceGroups source1)
    set(sources ${source1} ${ARGN})
    foreach(source ${sources})
        get_filename_component(source_path ${source} PATH)
        string(REPLACE "/" "\\" source_path_backslashes "${source_path}")
        source_group(${source_path_backslashes} FILES ${source})
    endforeach()
endfunction()

# Further processes a target and its list of source files adding extra touches useful for some generators
# (filter folders, group targets in folders, etc.).
# All optional arguments are treated as additional source files.
function(setup_target targetName source1)
    set(sources ${source1} ${ARGN})

    createSourceGroups(${sources})

    # Enable USE_FOLDERS. This is required by the set_target_properties(... FOLDER ...) call below.
    # We prefer to set it here rather than globally at the top of the file so that we only modify
    # the Cmake environment if/when the functionality is actually required.
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    file(RELATIVE_PATH projectFolder ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${targetName} PROPERTIES FOLDER "${projectFolder}")
endfunction()

# Convenience replacement of add_executable(), which besides adding an executable to the project
# further configures the target via setup_target().
# All optional arguments are treated as additional source files.
function(add_executable_ex targetName source1)
    set(sources ${source1} ${ARGN})
    add_executable(${targetName} ${sources})
    setup_target(${targetName} ${sources})
endfunction()

# Convenience replacement of add_library(), which besides adding a library to the project
# further configures the target via setup_target().
# All optional arguments are treated as additional source files.
function(add_library_ex targetName libraryType source1)
    set(sources ${source1} ${ARGN})
    add_library(${targetName} ${libraryType} ${sources})
    setup_target(${targetName} ${sources})
endfunction()
