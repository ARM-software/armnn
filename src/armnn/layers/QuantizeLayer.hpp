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
class ILayerVisitor;

class QuantizeLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    Layer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


protected:
    QuantizeLayer(const char* name);
    ~QuantizeLayer() = default;

};

} //namespace armnn
