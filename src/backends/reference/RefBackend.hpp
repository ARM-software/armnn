//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backends/IBackendInternal.hpp>
#include "RefLayerSupport.hpp"

namespace armnn
{

class RefBackend : public IBackendInternal
{
public:
    RefBackend()  = default;
    ~RefBackend() = default;

    const std::string& GetId() const override;

    const ILayerSupport& GetLayerSupport() const override;

    std::unique_ptr<IWorkloadFactory> CreateWorkloadFactory() const override;

    static void Destroy(IBackend* backend);

private:
    RefLayerSupport m_LayerSupport;
};

} // namespace armnn