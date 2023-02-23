//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ConstantLayer.hpp"
#include <Layer.hpp>

namespace armnn
{

template <typename Parameters>
class LayerWithParameters : public Layer
{
public:
    using DescriptorType = Parameters;

    const Parameters& GetParameters() const override { return m_Param; }

    /// Helper to serialize the layer parameters to string
    /// (currently used in DotSerializer and company).
    void SerializeLayerParameters(ParameterStringifyFunction& fn) const override
    {
        StringifyLayerParameters<Parameters>::Serialize(fn, m_Param);
        Layer::SerializeLayerParameters(fn);
    }

protected:
    LayerWithParameters(unsigned int numInputSlots,
                        unsigned int numOutputSlots,
                        LayerType type,
                        const Parameters& param,
                        const char* name)
        : Layer(numInputSlots, numOutputSlots, type, name)
        , m_Param(param)
    {
    }

    ~LayerWithParameters() = default;

    /// Helper function to reduce duplication in *Layer::CreateWorkload.
    template <typename QueueDescriptor>
    WorkloadInfo PrepInfoAndDesc(QueueDescriptor& descriptor) const
    {
        descriptor.m_Parameters = m_Param;
        descriptor.m_AllowExpandedDims = GetAllowExpandedDims();
        return Layer::PrepInfoAndDesc(descriptor);
    }

    /// The parameters for the layer (not including tensor-valued weights etc.).
    Parameters m_Param;

    void ExecuteStrategy(IStrategy& strategy) const override
    {
        strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
    }

    Layer::ImmutableConstantTensors GetConnectedConstantAsInputTensors() const
    {
        Layer::ImmutableConstantTensors tensors;
        for (unsigned int i = 0; i < GetNumInputSlots(); ++i)
        {
            if (GetInputSlot(i).GetConnection() && GetInputSlot(i).GetConnection()->GetTensorInfo().IsConstant())
            {
                auto &inputLayer = GetInputSlot(i).GetConnectedOutputSlot()->GetOwningLayer();
                if (inputLayer.GetType() == armnn::LayerType::Constant)
                {
                    auto &constantLayer = static_cast<ConstantLayer&>(inputLayer);

                    tensors.push_back(constantLayer.m_LayerOutput);
                }
            }
        }
        if (tensors.empty())
        {
            const std::string warningMessage{"GetConnectedConstantAsInputTensors() called on Layer with no "
                                             "connected Constants as Input Tensors."};
            ARMNN_LOG(warning) << warningMessage;
        }
        return tensors;
    }
};




} // namespace
