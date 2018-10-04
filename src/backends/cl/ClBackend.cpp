//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"

namespace armnn
{

const std::string ClBackend::s_Id = "arm_compute_cl";

const std::string& ClBackend::GetId() const
{
    return s_Id;
}

const ILayerSupport& ClBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> ClBackend::CreateWorkloadFactory() const
{
    // TODO implement
    return nullptr;
}

} // namespace armnn