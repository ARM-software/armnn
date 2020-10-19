//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateOptions.hpp>

namespace armnnDelegate
{

DelegateOptions::DelegateOptions(armnn::Compute computeDevice)
    : m_Backends({computeDevice})
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends)
    : m_Backends(backends)
{
}

} // namespace armnnDelegate
