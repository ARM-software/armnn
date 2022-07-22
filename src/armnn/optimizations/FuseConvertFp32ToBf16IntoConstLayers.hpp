//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"
#include <armnnUtils/Permute.hpp>
#include <ResolveType.hpp>

namespace armnn
{
namespace optimizations
{

class FuseConvertFp32ToBf16IntoConstLayers
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        ARMNN_ASSERT(base.GetType() == LayerType::Constant);
        ARMNN_ASSERT(child.GetType() == LayerType::ConvertFp32ToBf16);

        auto dataType = base.GetDataType();
        switch (dataType)
        {
            case DataType::Float32:
                ReplaceConvertFp32ToBf16Layer<DataType::BFloat16>(
                        graph,
                        PolymorphicDowncast<ConstantLayer*>(&base),
                        PolymorphicDowncast<ConvertFp32ToBf16Layer*>(&child));
                break;
            default:
                throw InvalidArgumentException(GetDataTypeName(dataType) +
                                               std::string(" Constant Layer cannot be fused into ")  +
                                               GetDataTypeName(child.GetDataType()) +
                                               std::string(" conversion layer."));
        }
    }
protected:
    FuseConvertFp32ToBf16IntoConstLayers()  = default;
    ~FuseConvertFp32ToBf16IntoConstLayers() = default;
private:
    template<armnn::DataType ArmnnType,
             typename T = armnn::ResolveType<ArmnnType>>
    static void ReplaceConvertFp32ToBf16Layer(Graph& graph,
                                              ConstantLayer* constantLayer,
                                              ConvertFp32ToBf16Layer* convertFp32ToBf16layer)
    {
        IgnoreUnused(graph);
        /**
         * This optimisation is to find situations where a constant set of inputs is being provided to a
         * ConvertFp32ToBf16 layer. In this case we don't want the overhead of Converting the values on
         * every inference, instead we want to Convert them once and store them in a Const layer to be
         * used everytime as they will not change.
         */
        TensorInfo outputConvertFp32ToBf16Info = convertFp32ToBf16layer->GetOutputSlot(0).GetTensorInfo();
        std::vector<T> newValues(outputConvertFp32ToBf16Info.GetNumElements());

        armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(
                constantLayer->m_LayerOutput->GetConstTensor<float>(),
                outputConvertFp32ToBf16Info.GetNumElements(),
                newValues.data());
        TensorInfo newInfo = outputConvertFp32ToBf16Info;
        newInfo.SetConstant(true);
        ConstTensor newInput(newInfo, newValues);

        constantLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));

        // Moves connections in convertFp32ToBf16layer output slot to the constant layer.
        // ConvertFp32ToBf16layer layer will be removed if left unconnected.
        convertFp32ToBf16layer->GetOutputSlot().MoveAllConnections(constantLayer->GetOutputSlot());

        // Updating the output tensor
        constantLayer->GetOutputSlot(0).SetTensorInfo(newInfo);
        ARMNN_ASSERT(constantLayer->GetOutputSlot(0).GetTensorInfo().IsConstant() == true);
    }
};

using FuseConversionLayersIntoConstLayers = OptimizeForConnection<ConstantLayer,
                                                                  ConvertFp32ToBf16Layer,
                                                                  FuseConvertFp32ToBf16IntoConstLayers>;

} // namespace optimizations
} // namespace armnn