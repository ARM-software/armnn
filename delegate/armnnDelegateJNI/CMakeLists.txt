#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.7.0)

project("armnn_delegate_jni")

# JNI is needed for jni calls
find_package(JNI)

list(APPEND jni_delegate_sources
        src/armnn_delegate_jni.cpp)

# the backends under src/backends extend the list of
# object libs armnn to include in the build
# If armnn is a static library (which it should be to make armnn_delegate_jni a stand alone library) then
# the object libraries of the backends need to be linked manually
include(${ARMNN_SOURCE_DIR}/src/backends/backends.cmake)
foreach(lib ${armnnLibraries})
    message(STATUS "Adding object library dependency to armnn_delegate_jni: ${lib}")
    list(APPEND jni_delegate_sources $<TARGET_OBJECTS:${lib}>)
endforeach()

if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
else()
    message (FATAL_ERROR "JNI library could not be found")
endif()
include_directories(${JNI_INCLUDE_DIRS})

add_library(armnn_delegate_jni SHARED ${jni_delegate_sources})

target_link_libraries(armnn_delegate_jni
        PRIVATE
            Armnn::Armnn
            ArmnnDelegate::ArmnnDelegate
        )

# A version script is used to hide all symbols that are not required to use the jni interface
# This is mostly required to avoid symbol conflicts between libc++_shared used to compile armnn
# and an eventual other version used somewhere else: https://developer.android.com/ndk/guides/cpp-support
#    This also requires to tell the compiler to link to the static version of libc++_shared. This can be accomplished
#    by adding -DCMAKE_ANDROID_STL_TYPE=c++_static to the cmake command when building for android
set(version_script "${CMAKE_CURRENT_SOURCE_DIR}/version_script")

# Generate a map file for debug mode only
set_property(TARGET armnn_delegate_jni APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,--version-script=${version_script},-Map=mapfile.map")

set_target_properties(armnn_delegate_jni PROPERTIES LINK_DEPENDS ${version_script})
