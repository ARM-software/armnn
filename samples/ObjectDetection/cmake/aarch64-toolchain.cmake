# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

# specify the cross compiler
set(GNU_MACHINE "aarch64-linux-gnu")
set(CROSS_PREFIX "aarch64-linux-gnu-")

set(CMAKE_C_COMPILER   ${CROSS_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_PREFIX}g++)
set(CMAKE_AR           ${CROSS_PREFIX}ar)
set(CMAKE_STRIP        ${CROSS_PREFIX}strip)
set(CMAKE_LINKER       ${CROSS_PREFIX}ld)

set(CMAKE_CROSSCOMPILING true)
set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(OPENCV_EXTRA_ARGS   "-DENABLE_NEON=ON"
                        "-DCMAKE_TOOLCHAIN_FILE=platforms/linux/aarch64-gnu.toolchain.cmake")