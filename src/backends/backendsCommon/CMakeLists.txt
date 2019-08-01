#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsCommon_sources
    BackendRegistry.cpp
    BackendRegistry.hpp
    CpuTensorHandle.cpp
    CpuTensorHandleFwd.hpp
    CpuTensorHandle.hpp
    DynamicBackend.cpp
    DynamicBackend.hpp
    DynamicBackendUtils.cpp
    DynamicBackendUtils.hpp
    IBackendInternal.cpp
    IBackendInternal.hpp
    IBackendContext.hpp
    ITensorHandleFactory.cpp
    ITensorHandleFactory.hpp
    LayerSupportBase.cpp
    LayerSupportBase.hpp
    LayerSupportRules.hpp
    IMemoryManager.hpp
    ITensorHandle.hpp
    MakeWorkloadHelper.hpp
    MemCopyWorkload.cpp
    MemCopyWorkload.hpp
    MemImportWorkload.cpp
    MemImportWorkload.hpp
    MemSyncWorkload.cpp
    MemSyncWorkload.hpp
    OptimizationViews.cpp
    OptimizationViews.hpp
    OutputHandler.cpp
    OutputHandler.hpp
    TensorHandleFactoryRegistry.cpp
    TensorHandleFactoryRegistry.hpp
    WorkloadDataCollector.hpp
    WorkloadData.cpp
    WorkloadDataFwd.hpp
    WorkloadData.hpp
    WorkloadFactory.cpp
    WorkloadFactory.hpp
    Workload.hpp
    WorkloadInfo.hpp
    WorkloadUtils.cpp
    WorkloadUtils.hpp
)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

add_library(armnnBackendsCommon OBJECT ${armnnBackendsCommon_sources})
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
