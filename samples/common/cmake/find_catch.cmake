# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

#Test TPIP
set(TEST_TPIP ${DEPENDENCIES_DIR}/test)
file(MAKE_DIRECTORY ${TEST_TPIP})
set(TEST_TPIP_INCLUDE ${TEST_TPIP}/include)
file(MAKE_DIRECTORY ${TEST_TPIP_INCLUDE})

ExternalProject_Add(catch2-headers
    URL https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/catch.hpp ${TEST_TPIP_INCLUDE}
    INSTALL_COMMAND ""
    )