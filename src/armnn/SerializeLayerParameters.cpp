//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "SerializeLayerParameters.hpp"
#include <armnn/TypesUtils.hpp>
#include <string>
#include <iostream>
#include <sstream>

namespace armnn
{

void
StringifyLayerParameters<PermuteDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                       const PermuteDescriptor & desc)
{
    std::stringstream ss;
    ss <<  "[";
    bool addComma = false;
    for (auto it=desc.m_DimMappings.begin(); it!= desc.m_DimMappings.end(); ++it)
    {
        if (addComma)
        {
            ss << ",";
        }
        ss << *it;
        addComma = true;
    }
    ss << "]";

    fn("DimMappings",ss.str());
}

void
StringifyLayerParameters<ReshapeDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                       const ReshapeDescriptor & desc)
{
    std::stringstream ss;
    ss <<  "[";
    bool addComma = false;
    for (unsigned int i=0; i<desc.m_TargetShape.GetNumDimensions(); ++i)
    {
        if (addComma)
        {
            ss << ",";
        }
        ss << desc.m_TargetShape[i];
        addComma = true;
    }
    ss << "]";

    fn("TargetShape",ss.str());
}

void
StringifyLayerParameters<ActivationDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                          const ActivationDescriptor & desc)
{
    fn("Function",GetActivationFunctionAsCString(desc.m_Function));
    fn("A",std::to_string(desc.m_A));
    fn("B",std::to_string(desc.m_B));
}

void
StringifyLayerParameters<Convolution2dDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                             const Convolution2dDescriptor & desc)
{
    {
        std::stringstream ss;
        ss << "(" << desc.m_PadTop    << "," << desc.m_PadLeft
           << "," << desc.m_PadBottom << "," << desc.m_PadRight << ")";
        fn("Padding(T,L,B,R)",ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_StrideX <<  "," << desc.m_StrideY << ")";
        fn("Stride(X,Y)", ss.str());
    }

    fn("BiasEnabled",(desc.m_BiasEnabled?"true":"false"));
}

void
StringifyLayerParameters<BatchNormalizationDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                                  const BatchNormalizationDescriptor & desc)
{
    fn("Eps",std::to_string(desc.m_Eps));
}

void
StringifyLayerParameters<DepthwiseConvolution2dDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                                      const DepthwiseConvolution2dDescriptor & desc)
{
    {
        std::stringstream ss;
        ss << "(" << desc.m_PadTop    << "," << desc.m_PadLeft
           << "," << desc.m_PadBottom << "," << desc.m_PadRight << ")";
        fn("Padding(T,L,B,R)",ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_StrideX <<  "," << desc.m_StrideY << ")";
        fn("Stride(X,Y)", ss.str());
    }

    fn("BiasEnabled",(desc.m_BiasEnabled?"true":"false"));
}

void
StringifyLayerParameters<Pooling2dDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                         const Pooling2dDescriptor & desc)
{
    fn("Type", GetPoolingAlgorithmAsCString(desc.m_PoolType));
    {
        std::stringstream ss;
        ss << "(" << desc.m_PadTop    << "," << desc.m_PadLeft
           << "," << desc.m_PadBottom << "," << desc.m_PadRight << ")";
        fn("Padding(T,L,B,R)",ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_PoolWidth    << "," << desc.m_PoolHeight << ")";
        fn("(Width,Height)",ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_StrideX <<  "," << desc.m_StrideY << ")";
        fn("Stride(X,Y)", ss.str());
    }

    fn("OutputShapeRounding", GetOutputShapeRoundingAsCString(desc.m_OutputShapeRounding));
    fn("PaddingMethod", GetPaddingMethodAsCString(desc.m_PaddingMethod));
}

void
StringifyLayerParameters<SoftmaxDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                       const SoftmaxDescriptor & desc)
{
    fn("Beta", std::to_string(desc.m_Beta));
}

void
StringifyLayerParameters<FullyConnectedDescriptor>::Serialize(ParameterStringifyFunction & fn,
                                                              const FullyConnectedDescriptor & desc)
{
    fn("BiasEnabled", (desc.m_BiasEnabled?"true":"false"));
    fn("TransposeWeightMatrix", (desc.m_TransposeWeightMatrix?"true":"false"));
}


}
