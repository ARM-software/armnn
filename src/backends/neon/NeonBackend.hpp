//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backends/IBackendInternal.hpp>
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

    static void Destroy(IBackend* backend);

private:
    NeonLayerSupport m_LayerSupport;
};

} // namespace armnn