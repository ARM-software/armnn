//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

namespace armnn {

//Forward
class IWorkload;
class IWorkloadFactory;

class QuantizeLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    Layer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

    void ExecuteStrategy(IStrategy& strategy) const override;


protected:
    QuantizeLayer(const char* name);
    ~QuantizeLayer() = default;

};

} //namespace armnn
