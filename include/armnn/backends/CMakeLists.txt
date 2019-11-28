#
# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsAPI_sources
     CpuTensorHandleFwd.hpp
     DynamicBackend.hpp
     IBackendInternal.hpp
     IBackendContext.hpp
     ITensorHandleFactory.hpp
     IMemoryManager.hpp
     ITensorHandle.hpp
     OptimizationViews.hpp
)

add_library(armnnBackendsAPI OBJECT ${armnnBackendsAPI_sources})
