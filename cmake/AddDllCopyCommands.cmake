#
# Copyright © 2018-2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
macro (addDllCopyCommand target sourceDebug sourceRelease)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<$<CONFIG:Debug>:${sourceDebug}>$<$<CONFIG:Release>:${sourceRelease}>$<$<CONFIG:RelWithDebInfo>:${sourceRelease}>$<$<CONFIG:MinSizeRel>:${sourceRelease}>"
        $<TARGET_FILE_DIR:${target}>)
endmacro()

# Checks if the given list contains an entry which matches the given regex.
function(listContainsRegex result list regex)
    set(${result} 0 PARENT_SCOPE)
    foreach(element ${list})
        if(${element} MATCHES ${regex})
            set(${result} 1 PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

macro(addDllCopyCommands target)
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        # Get the list of dependencies for the given target, so we can copy just the DLLs we need.
        get_target_property(target_deps_str ${target} LINK_LIBRARIES)
        set(target_deps)
        list(APPEND target_deps ${target_deps_str})

        cmake_policy(SET CMP0057 NEW) # Enable the "IN_LIST" operator

        # armnn.dll
        if ("armnn" IN_LIST target_deps)
            addDllCopyCommand(${target} "$<TARGET_FILE_DIR:armnn>/armnn.dll" "$<TARGET_FILE_DIR:armnn>/armnn.dll")
        endif()

        # armnnTfLiteParser.dll
        if ("armnnTfLiteParser" IN_LIST target_deps)
            addDllCopyCommand(${target} "$<TARGET_FILE_DIR:armnnTfLiteParser>/armnnTfLiteParser.dll"
                                        "$<TARGET_FILE_DIR:armnnTfLiteParser>/armnnTfLiteParser.dll")
        endif()
    endif()
endmacro()
