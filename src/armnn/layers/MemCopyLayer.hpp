//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class MemCopyLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload>
    CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const override;

    MemCopyLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    MemCopyLayer(const char* name);
    ~MemCopyLayer() = default;
};

} // namespace
