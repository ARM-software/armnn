#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsCommon_sources
    ClContextControl.cpp
    ClContextControl.hpp
    ClLayerSupport.cpp
    ClLayerSupport.hpp
    ClWorkloadFactory.cpp
    ClWorkloadFactory.hpp
    ClWorkloads.hpp
    CpuTensorHandle.cpp
    CpuTensorHandleFwd.hpp
    CpuTensorHandle.hpp
    ITensorHandle.hpp
    MakeWorkloadHelper.hpp
    MemCopyWorkload.cpp
    MemCopyWorkload.hpp
    NeonLayerSupport.cpp
    NeonLayerSupport.hpp
    NeonWorkloadFactory.cpp
    NeonWorkloadFactory.hpp
    NeonWorkloads.hpp
    OutputHandler.cpp
    OutputHandler.hpp
    RefLayerSupport.cpp
    RefLayerSupport.hpp
    RefWorkloadFactory.cpp
    RefWorkloadFactory.hpp
    RefWorkloads.hpp
    StringMapping.cpp
    StringMapping.hpp
    WorkloadDataCollector.hpp
    WorkloadData.cpp
    WorkloadDataFwd.hpp
    WorkloadData.hpp
    WorkloadFactory.cpp
    WorkloadFactory.hpp
    Workload.hpp
    WorkloadInfo.hpp
    WorkloadUtils.hpp
)

if(ARMCOMPUTENEON)
    # Additionally include source files for ARM Compute NEON backend
    list(APPEND armnnBackendsCommon_sources
        NeonWorkloadUtils.cpp
        NeonWorkloadUtils.hpp
        NeonTensorHandle.hpp
    )
endif()

if(ARMCOMPUTECL)
    # Additionally include source files for ARM Compute OpenCL backend
    list(APPEND armnnBackendsCommon_sources
        ClTensorHandle.hpp
    )
endif()

add_library(armnnBackendsCommon STATIC ${armnnBackendsCommon_sources})
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
