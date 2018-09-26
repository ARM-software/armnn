#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsCommon_sources
    CpuTensorHandle.cpp
    CpuTensorHandleFwd.hpp
    CpuTensorHandle.hpp
    ITensorHandle.hpp
    MakeWorkloadHelper.hpp
    MemCopyWorkload.cpp
    MemCopyWorkload.hpp
    OutputHandler.cpp
    OutputHandler.hpp
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

add_library(armnnBackendsCommon STATIC ${armnnBackendsCommon_sources})
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnBackendsCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
