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

class ConvertConstPermuteLayersToConstLayers
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        ARMNN_ASSERT(base.GetType() == LayerType::Constant);
        ARMNN_ASSERT(child.GetType() == LayerType::Permute);

        if (base.GetDataType() == child.GetDataType())
        {
            switch (base.GetDataType())
            {
                case DataType::Float16:
                    ReplaceConstPermuteLayer<DataType::Float16>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::Float32:
                    ReplaceConstPermuteLayer<DataType::Float32>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::QAsymmU8:
                    ReplaceConstPermuteLayer<DataType::QAsymmU8>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::Signed32:
                    ReplaceConstPermuteLayer<DataType::Signed32>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::QSymmS16:
                    ReplaceConstPermuteLayer<DataType::QSymmS16>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::QSymmS8:
                    ReplaceConstPermuteLayer<DataType::QSymmS8>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::QAsymmS8:
                    ReplaceConstPermuteLayer<DataType::QAsymmS8>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::BFloat16:
                    ReplaceConstPermuteLayer<DataType::BFloat16>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::Signed64:
                    ReplaceConstPermuteLayer<DataType::Signed64>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
                case DataType::Boolean:
                    ReplaceConstPermuteLayer<DataType::Boolean>(graph,
                                                                 PolymorphicDowncast<ConstantLayer*>(&base),
                                                                 PolymorphicDowncast<PermuteLayer*>(&child));
                    break;
            }
        }
    }
protected:
    ConvertConstPermuteLayersToConstLayers()  = default;
    ~ConvertConstPermuteLayersToConstLayers() = default;
private:
    template<armnn::DataType ArmnnType,
             typename T = armnn::ResolveType<ArmnnType>>
    static void ReplaceConstPermuteLayer(Graph& graph,
                                         ConstantLayer* constantLayer,
                                         PermuteLayer* permuteLayer)
    {
        IgnoreUnused(graph);
        /**
         * This optimisation is to find situations where a constant set of inputs is being provided to a Permute
         * layer. In this case we don't want the overhead of Permuting the values on every inference, instead we
         * want to Permute them once and store them in a Const layer to be used everytime as they will not change.
         */
        TensorInfo outputPermuteInfo = permuteLayer->GetOutputSlot(0).GetTensorInfo();
        std::vector<T> newValues(outputPermuteInfo.GetNumElements());
        armnnUtils::Permute(outputPermuteInfo.GetShape(), permuteLayer->GetPermutation(),
                            constantLayer->m_LayerOutput->Map(true), newValues.data(),
                            GetDataTypeSize(outputPermuteInfo.GetDataType()));

        TensorInfo newInfo = outputPermuteInfo;
        newInfo.SetConstant(true);
        ConstTensor newInput(newInfo, newValues);
        constantLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));

        // Moves connections in permute output to the constant layer.
        // Permute layer will be removed if left unconnected.
        permuteLayer->GetOutputSlot().MoveAllConnections(constantLayer->GetOutputSlot());

        // Updating the output tensor
        constantLayer->GetOutputSlot(0).SetTensorInfo(newInfo);
        ARMNN_ASSERT(constantLayer->GetOutputSlot(0).GetTensorInfo().IsConstant() == true);
    }
};

using FusePermuteIntoConstLayer = OptimizeForConnection<ConstantLayer,
                                                        PermuteLayer,
                                                        ConvertConstPermuteLayersToConstLayers>;

} // namespace optimizations
} // namespace armnn