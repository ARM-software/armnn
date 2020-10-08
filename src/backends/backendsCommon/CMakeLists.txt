#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsCommon_sources
    CpuTensorHandle.cpp
    CpuTensorHandle.hpp
    CpuTensorHandleFwd.hpp
    DynamicBackend.cpp
    DynamicBackend.hpp
    DynamicBackendUtils.cpp
    DynamicBackendUtils.hpp
    IBackendContext.hpp
    IBackendInternal.cpp
    IBackendInternal.hpp
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
    MemCopyWorkload.hpp
    MemImportWorkload.cpp
    MemImportWorkload.hpp
    MemSyncWorkload.cpp
    MemSyncWorkload.hpp
    OptimizationViews.cpp
    OptimizationViews.hpp
    TensorHandleFactoryRegistry.cpp
    TensorHandleFactoryRegistry.hpp
    UnmapWorkload.cpp
    UnmapWorkload.hpp
    WorkloadDataCollector.hpp
    Workload.hpp
    WorkloadData.cpp
    WorkloadData.hpp
    WorkloadFactory.cpp
    WorkloadFactory.hpp
    WorkloadFactoryBase.hpp
    WorkloadInfo.hpp
    WorkloadUtils.cpp
    WorkloadUtils.hpp
)

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