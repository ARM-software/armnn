//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateOptions.hpp>

namespace armnnDelegate
{

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const std::vector<armnn::BackendOptions>& backendOptions)
    : m_Backends({computeDevice}), m_BackendOptions(backendOptions)
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const std::vector<armnn::BackendOptions>& backendOptions)
    : m_Backends(backends), m_BackendOptions(backendOptions)
{
}

} // namespace armnnDelegate
