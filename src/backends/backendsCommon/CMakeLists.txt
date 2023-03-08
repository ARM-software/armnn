#
# Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(NOT BUILD_BARE_METAL AND NOT EXECUTE_NETWORK_STATIC)
list(APPEND armnnBackendsCommon_sources
    DynamicBackend.cpp
    DynamicBackend.hpp
    DynamicBackendUtils.cpp
    DynamicBackendUtils.hpp
)
endif()

list(APPEND armnnBackendsCommon_sources
    IBackendContext.hpp
    IBackendInternal.cpp
    IMemoryManager.hpp
    ITensorHandle.hpp
    ITensorHandleFactory.cpp
    ITensorHandleFactory.hpp
    LayerSupportBase.cpp
    LayerSupportBase.hpp
    LayerSupportRules.hpp
    MakeWorkloadHelper.hpp
    MapWorkload.cpp
    MapWorkload.hpp
    MemCopyWorkload.cpp
    MemImportWorkload.cpp
    MemImportWorkload.hpp
    MemoryManager.cpp
    MemoryManager.hpp
    MemSyncWorkload.cpp
    MemSyncWorkload.hpp
    OptimizationViews.cpp
    OptimizationViews.hpp
    TensorHandle.cpp
    TensorHandleFactoryRegistry.cpp
    TensorHandleFactoryRegistry.hpp
    UnmapWorkload.cpp
    UnmapWorkload.hpp
    WorkloadDataCollector.hpp
    WorkloadData.cpp
    WorkloadFactory.cpp
    WorkloadFactoryBase.hpp
    WorkloadInfo.hpp
    WorkloadUtils.cpp
    WorkloadUtils.hpp
)

add_subdirectory(memoryOptimizerStrategyLibrary)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

add_library(armnnBackendsCommon OBJECT ${armnnBackendsCommon_sources})
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/include/armnn/backends)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
