#
# Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(NOT DISABLE_DYNAMIC_BACKENDS)
list(APPEND armnnBackendsCommon_sources
    DynamicBackend.cpp
    DynamicBackendUtils.cpp
    DynamicBackendUtils.hpp
)
endif()

list(APPEND armnnBackendsCommon_sources
    IBackendInternal.cpp
    ITensorHandleFactory.cpp
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
    TensorHandle.cpp
    TensorHandleFactoryRegistry.cpp
    TensorHandleFactoryRegistry.hpp
    UnmapWorkload.cpp
    UnmapWorkload.hpp
    WorkloadDataCollector.hpp
    WorkloadData.cpp
    WorkloadFactory.cpp
    WorkloadFactoryBase.hpp
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
