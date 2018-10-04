//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"

namespace armnn
{

const std::string NeonBackend::s_Id = "arm_compute_neon";

const std::string& NeonBackend::GetId() const
{
    return s_Id;
}

const ILayerSupport& NeonBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> NeonBackend::CreateWorkloadFactory() const
{
    // TODO implement
    return nullptr;
}

} // namespace armnn