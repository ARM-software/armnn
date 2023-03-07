#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.7.0)
enable_language(ASM)
project(armnn_support_library)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17 -Wall -fexceptions -Werror -Wno-unused-parameter -Wno-unused-private-field -Wno-unused-variable -Wno-attributes -Wno-format-security -Wno-extern-c-compat -Wno-invalid-partial-specialization -Wno-unneeded-internal-declaration -Wno-unused-function -DNN_COMPATIBILITY_LIBRARY_BUILD -DNN_DEBUGGABLE")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -DOPENSSL_SMALL -DBORINGSSL_ANDROID_SYSTEM -DBORINGSSL_SHARED_LIBRARY -DBORINGSSL_IMPLEMENTATION")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

SET(libnnapi_support_include_directories)
list(APPEND libnnapi_support_include_directories
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/include/nnapi/
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/include/
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/include
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/include
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/shim_and_sl/public
        ${AOSP_MODULES_ROOT}/system/logging/liblog/include
        ${AOSP_MODULES_ROOT}/system/libbase/include
        ${AOSP_MODULES_ROOT}/frameworks/native/libs/nativewindow/include
        ${AOSP_MODULES_ROOT}/system/core/libcutils/include
        ${AOSP_MODULES_ROOT}/system/core/include
        ${AOSP_MODULES_ROOT}/external/tensorflow
        ${AOSP_MODULES_ROOT}/external/gemmlowp/
        ${AOSP_MODULES_ROOT}/external/ruy/
        ${AOSP_MODULES_ROOT}/external/eigen/
        ${AOSP_MODULES_ROOT}/external/boringssl/src/include
        ${AOSP_MODULES_ROOT}/external/boringssl/include)

include_directories(${libnnapi_support_include_directories})

set(base_sources)
list(APPEND base_sources
        ${AOSP_MODULES_ROOT}/system/libbase/abi_compatibility.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/posix_strerror_r.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/chrono_utils.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/cmsg.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/file.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/hex.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/logging.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/mapped_file.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/parsebool.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/parsenetaddress.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/process.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/properties.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/stringprintf.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/strings.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/test_utils.cpp
        ${AOSP_MODULES_ROOT}/system/libbase/threads.cpp)
add_library(base STATIC ${base_sources})
target_include_directories (base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (base PUBLIC ${libnnapi_support_include_directories})
target_compile_options(base PUBLIC -Wno-c99-designator)

file(GLOB TYPES_CL_SOURCE_FILES
    ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/*Validation.cpp
)

set(neuralnetworks_types_cl_sources)
list(APPEND neuralnetworks_types_cl_sources
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/DynamicCLDeps.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/SharedMemory.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/SharedMemoryAndroid.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/TypeUtils.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/OperationsValidationUtils.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/Types.cpp
        ${TYPES_CL_SOURCE_FILES}
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/Validation.cpp)

add_library(neuralnetworks_types_cl STATIC ${neuralnetworks_types_cl_sources})
target_include_directories (neuralnetworks_types_cl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (neuralnetworks_types_cl PUBLIC ${libnnapi_support_include_directories})

set(neuralnetworks_common_cl_sources)
list(APPEND neuralnetworks_common_cl_sources
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/BufferTracker.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/CpuExecutor.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/GraphDump.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/IndexedShapeWrapper.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/LegacyUtils.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/ModelUtils.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/MetaModel.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/OperationsExecutionUtils.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/types/src/OperationsUtils.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/TokenHasher.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/OperationResolver.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ActivationExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/BatchMatmulExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/BidirectionalSequenceRNNExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/BroadcastExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ChannelShuffleExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ComparisonsExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ConcatenationExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/Conv2DExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/DepthwiseConv2DExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/DequantizeExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ElementwiseExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/EluExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/FillExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/FullyConnectedExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/GatherExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/GenerateProposalsExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/HeatmapMaxKeypointExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/InstanceNormalizationExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/L2NormalizationExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/LocalResponseNormalizationExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/LogSoftmaxExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/LogicalAndOrExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/LogicalNotExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/MirrorPadExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/NegExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/PackExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/PReluExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/PoolingExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/QLSTMExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/QuantizeExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/RankExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ReduceExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ReshapeExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ReverseExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/ResizeImageOpsExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/RoiAlignExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/RoiPoolingExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/SelectExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/SliceExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/SoftmaxExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/SqueezeExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/StridedSliceExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/TopK_V2Execution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/TransposeExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/TransposeConv2DExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/UnidirectionalSequenceLSTMExecution.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/common/operations/UnidirectionalSequenceRNNExecution.cpp)

add_library(neuralnetworks_common_cl STATIC ${neuralnetworks_common_cl_sources})
target_include_directories (neuralnetworks_common_cl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (neuralnetworks_common_cl PUBLIC ${libnnapi_support_include_directories})
target_compile_options(neuralnetworks_common_cl PUBLIC -Wno-unused-but-set-variable)

set(neuralnetworks_cl_sources)
list(APPEND neuralnetworks_cl_sources
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/BurstBuilder.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/CompilationBuilder.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ExecutionBuilder.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ExecutionCallback.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ExecutionPlan.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ServerFlag.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/Manager.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/Memory.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ModelArchHasher.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ModelArgumentInfo.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/ModelBuilder.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/NeuralNetworks.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/SupportLibraryDiagnostic.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/Telemetry.cpp
        ${AOSP_MODULES_ROOT}/packages/modules/NeuralNetworks/runtime/TypeManager.cpp)

add_library(neuralnetworks_cl OBJECT ${neuralnetworks_cl_sources})
target_include_directories (neuralnetworks_cl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (neuralnetworks_cl PUBLIC ${libnnapi_support_include_directories})

set(crypto_static_sources)
list(APPEND crypto_static_sources
        ${AOSP_MODULES_ROOT}/external/boringssl/err_data.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_bitstr.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_bool.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_d2i_fp.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_dup.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_enum.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_gentm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_i2d_fp.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_int.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_mbstr.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_object.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_octet.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_print.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_strex.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_strnid.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_time.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_type.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_utctm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/a_utf8.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/asn1_lib.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/asn1_par.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/asn_pack.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/f_int.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/f_string.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/tasn_dec.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/tasn_enc.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/tasn_fre.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/tasn_new.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/tasn_typ.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/tasn_utl.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/asn1/time_support.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/base64/base64.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/bio.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/bio_mem.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/connect.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/fd.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/file.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/hexdump.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/pair.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/printf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/socket.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bio/socket_helper.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/blake2/blake2.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bn_extra/bn_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bn_extra/convert.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/buf/buf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bytestring/asn1_compat.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bytestring/ber.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bytestring/cbb.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bytestring/cbs.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bytestring/unicode.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/bytestring/unicode.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/chacha/chacha.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/cipher_extra.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/derive_key.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_aesccm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_aesctrhmac.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_aesgcmsiv.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_chacha20poly1305.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_des.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_null.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_rc2.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_rc4.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_tls.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/tls_cbc.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cmac/cmac.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/conf/conf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_apple.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_fuchsia.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_linux.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_win.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_arm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_arm_linux.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_intel.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_ppc64le.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/crypto.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/curve25519/curve25519.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/curve25519/spake25519.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/des/des.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dh_extra/dh_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dh_extra/params.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/digest_extra/digest_extra.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dsa/dsa.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dsa/dsa_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ec_extra/ec_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ec_extra/ec_derive.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_aesccm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_aesctrhmac.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_aesgcmsiv.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_chacha20poly1305.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_des.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_null.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_rc2.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_rc4.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/e_tls.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cipher_extra/tls_cbc.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cmac/cmac.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/conf/conf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_apple.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_fuchsia.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_linux.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_aarch64_win.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_arm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_arm_linux.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_intel.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/cpu_ppc64le.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/crypto.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/curve25519/curve25519.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/curve25519/spake25519.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/des/des.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dh_extra/dh_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dh_extra/params.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/digest_extra/digest_extra.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dsa/dsa.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/dsa/dsa_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ec_extra/ec_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ec_extra/ec_derive.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ec_extra/hash_to_curve.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ecdh_extra/ecdh_extra.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ecdsa_extra/ecdsa_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/engine/engine.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/err/err.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/digestsign.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/evp.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/evp_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/evp_ctx.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_dsa_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_ec.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_ec_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_ed25519.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_ed25519_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_rsa.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_rsa_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_x25519.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/p_x25519_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/pbkdf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/print.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/scrypt.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/evp/sign.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/ex_data.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/fipsmodule/fips_shared_support.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/hkdf/hkdf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/hpke/hpke.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/hrss/hrss.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/lhash/lhash.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/mem.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/obj/obj.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/obj/obj_xref.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_all.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_info.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_lib.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_oth.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_pk8.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_pkey.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_x509.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pem/pem_xaux.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pkcs7/pkcs7.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pkcs7/pkcs7_x509.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pkcs8/p5_pbev2.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pkcs8/pkcs8.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pkcs8/pkcs8_x509.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/poly1305/poly1305.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/poly1305/poly1305_arm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/poly1305/poly1305_vec.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/pool/pool.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rand_extra/deterministic.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rand_extra/forkunsafe.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rand_extra/fuchsia.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rand_extra/passive.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rand_extra/rand_extra.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rand_extra/windows.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rc4/rc4.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/refcount_c11.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/refcount_lock.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rsa_extra/rsa_asn1.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/rsa_extra/rsa_print.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/siphash/siphash.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/stack/stack.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/thread.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/thread_none.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/thread_pthread.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/thread_win.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/trust_token/pmbtoken.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/trust_token/trust_token.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/trust_token/voprf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/a_digest.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/a_sign.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/a_verify.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/algorithm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/asn1_gen.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/by_dir.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/by_file.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/i2d_pr.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/name_print.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/rsa_pss.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/t_crl.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/t_req.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/t_x509.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/t_x509a.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_att.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_cmp.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_d2.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_def.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_ext.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_lu.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_obj.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_req.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_set.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_trs.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_txt.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_v3.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_vfy.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509_vpm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509cset.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509name.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509rset.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x509spki.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_algor.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_all.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_attrib.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_crl.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_exten.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_info.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_name.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_pkey.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_pubkey.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_req.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_sig.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_spki.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_val.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_x509.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509/x_x509a.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/pcy_cache.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/pcy_data.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/pcy_lib.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/pcy_map.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/pcy_node.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/pcy_tree.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_akey.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_akeya.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_alt.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_bcons.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_bitst.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_conf.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_cpols.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_crld.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_enum.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_extku.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_genn.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_ia5.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_info.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_int.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_lib.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_ncons.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_ocsp.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_pci.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_pcia.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_pcons.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_pmaps.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_prn.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_purp.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_skey.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/x509v3/v3_utl.c
        ${AOSP_MODULES_ROOT}/external/boringssl/src/crypto/fipsmodule/bcm.c
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/chacha/chacha-armv8.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/aesv8-armx64.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/armv8-mont.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/ghash-neon-armv8.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/ghashv8-armx64.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/sha1-armv8.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/sha256-armv8.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/sha512-armv8.S
        ${AOSP_MODULES_ROOT}/external/boringssl/linux-aarch64/crypto/fipsmodule/vpaes-armv8.S
        )

add_library(crypto_static STATIC ${crypto_static_sources})
target_include_directories (crypto_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (crypto_static PUBLIC ${libnnapi_support_include_directories})

message(STATUS "AOSP_MODULES_ROOT: ${AOSP_MODULES_ROOT}")

# Add Armnn as a Dependency
message(STATUS "ARMNN_SOURCE_DIR: ${ARMNN_SOURCE_DIR}")
message(STATUS "Armnn_DIR: ${Armnn_DIR}")

set(Armnn_DIR "${Armnn_DIR}")
if(NOT ARMNN_SUB_PROJECT)
    find_package(Armnn REQUIRED CONFIG HINTS ${Armnn_DIR})
endif()

add_library(thirdparty_headers INTERFACE)
target_include_directories(thirdparty_headers INTERFACE $<BUILD_INTERFACE:${ARMNN_SOURCE_DIR}/third-party>
        $<INSTALL_INTERFACE:include/thirdparty_headers>)

add_library(profiling_library_headers INTERFACE)
target_include_directories(profiling_library_headers INTERFACE $<BUILD_INTERFACE:${ARMNN_SOURCE_DIR}/profiling>
        $<INSTALL_INTERFACE:include/profiling_library_headers>)

set(armnn_support_library_sources)
list(APPEND armnn_support_library_sources
        canonical/ArmnnPreparedModel.cpp
        canonical/ArmnnPreparedModel.hpp
        canonical/ArmnnDevice.cpp
        canonical/ArmnnDevice.hpp
        canonical/ArmnnDriver.hpp
        canonical/ArmnnDriverImpl.cpp
        canonical/ArmnnDriverImpl.hpp
        canonical/CanonicalUtils.cpp
        canonical/CanonicalUtils.hpp
        canonical/ConversionUtils.cpp
        canonical/ConversionUtils.hpp
        canonical/Converter.cpp
        canonical/Converter.hpp
        canonical/DriverOptions.cpp
        canonical/DriverOptions.hpp
        canonical/ModelToINetworkTransformer.cpp
        canonical/ModelToINetworkTransformer.hpp
        canonical/SystemPropertiesUtils.hpp
        support_library_service.cpp)

add_library(armnn_support_library SHARED ${armnn_support_library_sources})
target_link_libraries(armnn_support_library PUBLIC Armnn::armnnUtils)
target_link_libraries(armnn_support_library PUBLIC Armnn::armnnSerializer-static)

target_link_libraries(armnn_support_library PUBLIC Armnn::Armnn)
target_link_libraries(armnn_support_library PUBLIC profiling_library_headers)
target_link_libraries(armnn_support_library PRIVATE thirdparty_headers)

target_link_libraries(armnn_support_library PRIVATE neuralnetworks_types_cl)
target_link_libraries(armnn_support_library PRIVATE neuralnetworks_common_cl)
target_link_libraries(armnn_support_library PRIVATE neuralnetworks_cl)
target_link_libraries(armnn_support_library PRIVATE -Wl, -Bsymbolic, crypto_static)
target_link_libraries(armnn_support_library PRIVATE base)

target_include_directories(armnn_support_library PUBLIC ${libnnapi_support_include_directories})

target_link_libraries(armnn_support_library PUBLIC -Wl,-undefined -Wl,dynamic_lookup)

# find the liblog
find_library(log-lib log)
target_link_libraries(armnn_support_library PUBLIC ${log-lib})

# find the libnativewindow
find_library(nativewindow-lib nativewindow)
target_link_libraries(armnn_support_library PUBLIC ${nativewindow-lib})

####################################################
