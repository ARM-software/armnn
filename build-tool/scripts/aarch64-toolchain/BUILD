# Copyright © 2026 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

load("@bazel_tools//tools/cpp:unix_cc_toolchain_config.bzl", "cc_toolchain_config")
load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")

package(default_visibility = ["//visibility:public"])

# Match the Debian/Ubuntu cross compiler used by CMake and ACL in common.sh.
# Let GCC select its own headers and sysroot so all dependencies share an ABI.
cc_toolchain_config(
    name = "config",
    cpu = "aarch64",
    compiler = "gcc",
    toolchain_identifier = "aarch64-linux-gnu",
    host_system_name = "local",
    target_system_name = "aarch64-linux-gnu",
    target_libc = "glibc",
    abi_version = "gcc",
    abi_libc_version = "glibc",
    cxx_builtin_include_directories = [
        "/usr/aarch64-linux-gnu/include",
        "/usr/lib/gcc-cross/aarch64-linux-gnu",
    ],
    tool_paths = {
        "ar": "/usr/bin/aarch64-linux-gnu-ar",
        "cpp": "/usr/bin/aarch64-linux-gnu-cpp",
        "dwp": "/usr/bin/aarch64-linux-gnu-dwp",
        "gcc": "/usr/bin/aarch64-linux-gnu-gcc",
        "gcov": "/usr/bin/aarch64-linux-gnu-gcov",
        "ld": "/usr/bin/aarch64-linux-gnu-ld",
        "llvm-cov": "/bin/false",
        "nm": "/usr/bin/aarch64-linux-gnu-nm",
        "objcopy": "/usr/bin/aarch64-linux-gnu-objcopy",
        "objdump": "/usr/bin/aarch64-linux-gnu-objdump",
        "strip": "/usr/bin/aarch64-linux-gnu-strip",
    },
    compile_flags = ["-fstack-protector"],
    opt_compile_flags = ["-O2", "-DNDEBUG", "-ffunction-sections", "-fdata-sections"],
    dbg_compile_flags = ["-g"],
    # Keep Bazel's symlinked dependency headers relative in GCC's dependency files.
    unfiltered_compile_flags = ["-fno-canonical-system-headers", "-no-canonical-prefixes"],
    link_libs = ["-lstdc++", "-lm"],
)

filegroup(name = "empty")

cc_toolchain(
    name = "cc-compiler-aarch64",
    toolchain_identifier = "aarch64-linux-gnu",
    toolchain_config = ":config",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "aarch64": ":cc-compiler-aarch64",
        "aarch64|gcc": ":cc-compiler-aarch64",
    },
)
