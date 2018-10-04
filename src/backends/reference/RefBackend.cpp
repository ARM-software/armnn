//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"

namespace armnn
{

const std::string RefBackend::s_Id = "arm_reference";

const std::string& RefBackend::GetId() const
{
    return s_Id;
}

const ILayerSupport& RefBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> RefBackend::CreateWorkloadFactory() const
{
    // TODO implement
    return nullptr;
}

} // namespace armnn