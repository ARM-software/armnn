macro (addDllCopyCommand target sourceDebug sourceRelease)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<$<CONFIG:Debug>:${sourceDebug}>$<$<CONFIG:Release>:${sourceRelease}>$<$<CONFIG:RelWithDebInfo>:${sourceRelease}>$<$<CONFIG:MinSizeRel>:${sourceRelease}>"
        $<TARGET_FILE_DIR:${target}>)
endmacro()

macro (addBoostDllCopyCommand target ignored sourceReleaseLib ignored sourceDebugLib)
    string(REGEX REPLACE ".lib$" ".dll" sourceReleaseDll ${sourceReleaseLib})
    string(REGEX REPLACE "/libboost" "/boost" sourceReleaseDll2 ${sourceReleaseDll})

    string(REGEX REPLACE ".lib$" ".dll" sourceDebugDll ${sourceDebugLib})
    string(REGEX REPLACE "/libboost" "/boost" sourceDebugDll2 ${sourceDebugDll})
    addDllCopyCommand(${target} ${sourceDebugDll2} ${sourceReleaseDll2})
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

        # armnnCaffeParser.dll
        if ("armnnCaffeParser" IN_LIST target_deps)
            addDllCopyCommand(${target} "$<TARGET_FILE_DIR:armnnCaffeParser>/armnnCaffeParser.dll"
                                        "$<TARGET_FILE_DIR:armnnCaffeParser>/armnnCaffeParser.dll")
            addDllCopyCommand(${target} "${PROTOBUF_ROOT}/bin/libprotobufd.dll"
                                        "${PROTOBUF_ROOT}/bin/libprotobuf.dll")
       endif()

        # armnnTfParser.dll
        if ("armnnTfParser" IN_LIST target_deps)
            addDllCopyCommand(${target} "$<TARGET_FILE_DIR:armnnTfParser>/armnnTfParser.dll"
                                        "$<TARGET_FILE_DIR:armnnTfParser>/armnnTfParser.dll")
            addDllCopyCommand(${target} "${PROTOBUF_ROOT}/bin/libprotobufd.dll"
                                        "${PROTOBUF_ROOT}/bin/libprotobuf.dll")
        endif()

        # armnnTfLiteParser.dll
        if ("armnnTfLiteParser" IN_LIST target_deps)
            addDllCopyCommand(${target} "$<TARGET_FILE_DIR:armnnTfLiteParser>/armnnTfLiteParser.dll"
                                        "$<TARGET_FILE_DIR:armnnTfLiteParser>/armnnTfLiteParser.dll")
        endif()

        # caffe.dll and its dependencies
        listContainsRegex(includeCaffeDlls "${target_deps}" "caffe")
        if (${includeCaffeDlls})
            addDllCopyCommand(${target} "${CAFFE_BUILD_ROOT}/lib/caffe-d.dll"
                                        "${CAFFE_BUILD_ROOT}/lib/caffe.dll")
            addDllCopyCommand(${target} "${PROTOBUF_ROOT}/bin/libprotobufd.dll"
                                        "${PROTOBUF_ROOT}/bin/libprotobuf.dll")
            addDllCopyCommand(${target} "${BLAS_ROOT}/bin/libopenblas.dll"          "${BLAS_ROOT}/bin/libopenblas.dll")
            addDllCopyCommand(${target} "${MINGW32_ROOT}/bin/libgfortran-3.dll"     "${MINGW32_ROOT}/bin/libgfortran-3.dll")
            addDllCopyCommand(${target} "${MINGW32_ROOT}/bin/libgcc_s_dw2-1.dll"    "${MINGW32_ROOT}/bin/libgcc_s_dw2-1.dll")
            addDllCopyCommand(${target} "${MINGW32_ROOT}/bin/libquadmath-0.dll"     "${MINGW32_ROOT}/bin/libquadmath-0.dll")
        endif()
    endif()
endmacro()
