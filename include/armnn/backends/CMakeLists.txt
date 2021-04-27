#
# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsAPI_sources
     CpuTensorHandleFwd.hpp
     TensorHandleFwd.hpp
     DynamicBackend.hpp
     IBackendInternal.hpp
     IBackendContext.hpp
     IMemoryManager.hpp
     ITensorHandle.hpp
     ITensorHandleFactory.hpp
     IWorkload.hpp
     OptimizationViews.hpp
     WorkloadInfo.hpp
     profiling/IBackendProfiling.hpp
     profiling/IBackendProfilingContext.hpp
)

add_library(armnnBackendsAPI OBJECT ${armnnBackendsAPI_sources})
