//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "backends/IBackendInternal.hpp"

#include "NeonLayerSupport.hpp"

namespace armnn
{

class NeonBackend : public IBackendInternal
{
public:
    NeonBackend()  = default;
    ~NeonBackend() = default;

    const std::string& GetId() const override;

    const ILayerSupport& GetLayerSupport() const override;

    std::unique_ptr<IWorkloadFactory> CreateWorkloadFactory() const override;

private:
    static const std::string s_Id;

    // TODO initialize
    NeonLayerSupport m_LayerSupport;
};

} // namespace armnn