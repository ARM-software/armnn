#
# Copyright © 2020, 2023-2025 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(TFLITE_FOUND)

#
# NOTE: this module is used to find the tensorflow lite binary libraries only
#       the FindTfLiteSrc.cmake module is used to find the tensorflow lite include directory.
#       This is to allow components like the Tensorflow lite parser that have a source dependency
#       on tensorflow lite headers but no need to link to the binary libraries to use only the sources
#       and not have an artificial dependency on the libraries.
#
# First look for the static version of tensorflow lite
find_library(TfLite_LIB NAMES "libtensorflow-lite.a" HINTS ${TFLITE_LIB_ROOT} ${TFLITE_LIB_ROOT}/tensorflow/lite NO_CMAKE_FIND_ROOT_PATH  )
# If not found then, look for the dynamic library of tensorflow lite
find_library(TfLite_LIB NAMES "libtensorflow_lite_all.so" "libtensorflowlite.so" HINTS ${TFLITE_LIB_ROOT} ${TFLITE_LIB_ROOT}/tensorflow/lite NO_CMAKE_FIND_ROOT_PATH)

# If the static library was found, gather all of its dependencies
if (TfLite_LIB MATCHES .a$)
    message("-- Static tensorflow lite library found, using for ArmNN build")
    find_library(TfLite_abseilstrings_LIB "libabsl_strings.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/strings
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_internal_strings_LIB "libabsl_strings_internal.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/strings
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_synchronization_LIB "libabsl_synchronization.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/synchronization
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_kernel_timeout_LIB "libabsl_kernel_timeout_internal.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/synchronization
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_raw_logging_internal_LIB "libabsl_raw_logging_internal.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/base
                NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    # Required for building TensorFlow in Debug
    find_library(TfLite_abseil_graphCycle_internal_LIB "libabsl_graphcycles_internal.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/synchronization NO_CMAKE_FIND_ROOT_PATH )
    find_library(TfLite_farmhash_LIB "libfarmhash.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/farmhash-build NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_fftsg_LIB "libfft2d_fftsg.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/fft2d-build NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_fftsg2d_LIB "libfft2d_fftsg2d.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/fft2d-build NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_flatbuffers_LIB "libflatbuffers.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/flatbuffers-build
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_cpuinfo_LIB "libcpuinfo.a" PATH
                 ${TFLITE_LIB_ROOT}/_deps/cpuinfo-build NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_allocator_LIB "libruy_allocator.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_apply_multiplier_LIB "libruy_apply_multiplier.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_blocking_counter_LIB "libruy_blocking_counter.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_block_map_LIB "libruy_block_map.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_context_LIB "libruy_context.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_context_get_ctx_LIB "libruy_context_get_ctx.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_cpuinfo_LIB "libruy_cpuinfo.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_ctx_LIB "libruy_ctx.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_denormal_LIB "libruy_denormal.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_frontend_LIB "libruy_frontend.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_have_built_path_for_avx2_fma_LIB "libruy_have_built_path_for_avx2_fma.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_have_built_path_for_avx512_LIB "libruy_have_built_path_for_avx512.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_have_built_path_for_avx_LIB "libruy_have_built_path_for_avx.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_kernel_arm_LIB "libruy_kernel_arm.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_kernel_avx2_fma_LIB "libruy_kernel_avx2_fma.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_kernel_avx512_LIB "libruy_kernel_avx512.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_kernel_avx_LIB "libruy_kernel_avx.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_pack_arm_LIB "libruy_pack_arm.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_pack_avx2_fma_LIB "libruy_pack_avx2_fma.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_pack_avx512_LIB "libruy_pack_avx512.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_pack_avx_LIB "libruy_pack_avx.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_prepacked_cache_LIB "libruy_prepacked_cache.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_prepare_packed_matrices_LIB "libruy_prepare_packed_matrices.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_system_aligned_alloc_LIB "libruy_system_aligned_alloc.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_threadpool_LIB "libruy_thread_pool.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_trmul_LIB "libruy_trmul.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_tune_LIB "libruy_tune.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_wait_LIB "libruy_wait.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_ruy_profiler_LIB "libruy_profiler_instrumentation.a"
                 PATHS ${TFLITE_LIB_ROOT}/_deps/ruy-build/ruy/profiler
                 NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    ## Set TFLITE_FOUND if all libraries are satisfied for static lib
    find_package_handle_standard_args(TfLite DEFAULT_MSG TfLite_LIB TfLite_abseilstrings_LIB TfLite_farmhash_LIB TfLite_fftsg_LIB TfLite_fftsg2d_LIB
                                      TfLite_flatbuffers_LIB TfLite_ruy_allocator_LIB TfLite_ruy_apply_multiplier_LIB TfLite_ruy_blocking_counter_LIB
                                      TfLite_ruy_block_map_LIB TfLite_ruy_context_LIB TfLite_ruy_context_get_ctx_LIB TfLite_ruy_cpuinfo_LIB
                                      TfLite_ruy_ctx_LIB TfLite_ruy_denormal_LIB TfLite_ruy_frontend_LIB TfLite_ruy_have_built_path_for_avx2_fma_LIB
                                      TfLite_ruy_have_built_path_for_avx512_LIB TfLite_ruy_have_built_path_for_avx_LIB TfLite_ruy_kernel_arm_LIB
                                      TfLite_ruy_kernel_avx2_fma_LIB TfLite_ruy_kernel_avx512_LIB TfLite_ruy_kernel_avx_LIB TfLite_ruy_pack_arm_LIB
                                      TfLite_ruy_pack_avx2_fma_LIB TfLite_ruy_pack_avx512_LIB TfLite_ruy_pack_avx_LIB TfLite_ruy_prepacked_cache_LIB
                                      TfLite_ruy_prepare_packed_matrices_LIB TfLite_ruy_system_aligned_alloc_LIB TfLite_ruy_threadpool_LIB
                                      TfLite_ruy_trmul_LIB TfLite_ruy_tune_LIB TfLite_ruy_wait_LIB TfLite_ruy_profiler_LIB TfLite_cpuinfo_LIB
                                      TfLite_abseil_synchronization_LIB TfLite_abseil_graphCycle_internal_LIB TfLite_abseil_raw_logging_internal_LIB
                                      TfLite_abseil_kernel_timeout_LIB TfLite_abseil_internal_strings_LIB)
    # Set external variables for usage in CMakeLists.txt
    if (TFLITE_FOUND)
        # WARNING! The order of these libraries is critical. Moving them
        # around will result in linker errors in DelegateUnitTests.
        set(TfLite_LIB ${TfLite_LIB} ${TfLite_abseilstrings_LIB} ${TfLite_farmhash_LIB} ${TfLite_fftsg_LIB} ${TfLite_fftsg2d_LIB} ${TfLite_flatbuffers_LIB}
                                     ${TfLite_ruy_allocator_LIB} ${TfLite_ruy_apply_multiplier_LIB} ${TfLite_ruy_frontend_LIB} ${TfLite_ruy_trmul_LIB}
                                     ${TfLite_ruy_threadpool_LIB} ${TfLite_ruy_blocking_counter_LIB} ${TfLite_ruy_block_map_LIB} ${TfLite_ruy_context_LIB}
                                     ${TfLite_ruy_context_get_ctx_LIB} ${TfLite_ruy_cpuinfo_LIB} ${TfLite_ruy_ctx_LIB} ${TfLite_ruy_denormal_LIB}
                                     ${TfLite_ruy_have_built_path_for_avx2_fma_LIB} ${TfLite_ruy_have_built_path_for_avx512_LIB}
                                     ${TfLite_ruy_have_built_path_for_avx_LIB} ${TfLite_ruy_kernel_arm_LIB} ${TfLite_ruy_kernel_avx2_fma_LIB}
                                     ${TfLite_ruy_kernel_avx512_LIB} ${TfLite_ruy_kernel_avx_LIB} ${TfLite_ruy_pack_arm_LIB}
                                     ${TfLite_ruy_pack_avx2_fma_LIB} ${TfLite_ruy_pack_avx512_LIB} ${TfLite_ruy_pack_avx_LIB} ${TfLite_ruy_prepacked_cache_LIB}
                                     ${TfLite_ruy_prepare_packed_matrices_LIB} ${TfLite_ruy_system_aligned_alloc_LIB}
                                     ${TfLite_ruy_tune_LIB} ${TfLite_ruy_wait_LIB} ${TfLite_ruy_profiler_LIB}
                                     ${TfLite_cpuinfo_LIB} ${TfLite_abseil_synchronization_LIB} ${TfLite_abseil_graphCycle_internal_LIB}
                                     ${TfLite_abseil_raw_logging_internal_LIB} ${TfLite_abseil_kernel_timeout_LIB} ${TfLite_abseil_internal_strings_LIB})
    endif ()
elseif (TfLite_LIB MATCHES .so$)
    message("-- Dynamic tensorflow lite library found, using for ArmNN build")
    find_package_handle_standard_args(TfLite DEFAULT_MSG TfLite_LIB)
    ## Set external variables for usage in CMakeLists.txt
    if (TFLITE_FOUND)
        set(TfLite_LIB ${TfLite_LIB})
    endif ()
else()
    message(FATAL_ERROR "Could not find a tensorflow lite library to use")
endif()
