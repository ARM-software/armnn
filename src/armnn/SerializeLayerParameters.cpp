//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SerializeLayerParameters.hpp"
#include <armnn/TypesUtils.hpp>
#include <string>
#include <iostream>
#include <sstream>

namespace armnn
{

void StringifyLayerParameters<ActivationDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                               const ActivationDescriptor& desc)
{
    fn("Function", GetActivationFunctionAsCString(desc.m_Function));
    fn("A", std::to_string(desc.m_A));
    fn("B", std::to_string(desc.m_B));
}

void StringifyLayerParameters<BatchNormalizationDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                       const BatchNormalizationDescriptor& desc)
{
    fn("Eps", std::to_string(desc.m_Eps));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<BatchToSpaceNdDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                   const BatchToSpaceNdDescriptor& desc)
{
    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_BlockShape)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("BlockShape", ss.str());
    }

    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_Crops)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << "[" << var.first << "," << var.second << "]";
            ++count;
        }
        fn("Crops", ss.str());
    }

    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<ChannelShuffleDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                   const ChannelShuffleDescriptor& desc)
{
    fn("Axis", std::to_string(desc.m_Axis));
    fn("NumGroups", std::to_string(desc.m_NumGroups));
}

void StringifyLayerParameters<ComparisonDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                               const ComparisonDescriptor& desc)
{
    fn("Operation", GetComparisonOperationAsCString(desc.m_Operation));
}

void StringifyLayerParameters<Convolution2dDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                  const Convolution2dDescriptor& desc)
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

    {
        std::stringstream ss;
        ss << "(" << desc.m_DilationX << "," << desc.m_DilationY << ")";
        fn("Dilation(X,Y)", ss.str());
    }

    fn("BiasEnabled",(desc.m_BiasEnabled ? "true" : "false"));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<Convolution3dDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                  const Convolution3dDescriptor& desc)
{
    {
        std::stringstream ss;
        ss << "(" << desc.m_PadTop    << "," << desc.m_PadLeft
           << "," << desc.m_PadBottom << "," << desc.m_PadRight
           << "," << desc.m_PadFront  << "," << desc.m_PadBack << ")";
        fn("Padding(T,L,B,R,F,B)",ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_StrideX <<  "," << desc.m_StrideY << "," << desc.m_StrideZ << ")";
        fn("Stride(X,Y,Z)", ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_DilationX << "," << desc.m_DilationY << "," << desc.m_DilationZ << ")";
        fn("Dilation(X,Y,Z)", ss.str());
    }

    fn("BiasEnabled",(desc.m_BiasEnabled ? "true" : "false"));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<DetectionPostProcessDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                         const DetectionPostProcessDescriptor& desc)
{
    fn("MaxDetections", std::to_string(desc.m_MaxDetections));
    fn("MaxClassesPerDetection", std::to_string(desc.m_MaxClassesPerDetection));
    fn("DetectionsPerClass", std::to_string(desc.m_DetectionsPerClass));
    fn("NmsScoreThreshold", std::to_string(desc.m_NmsScoreThreshold));
    fn("NmsIouThreshold", std::to_string(desc.m_NmsIouThreshold));
    fn("NumClasses", std::to_string(desc.m_NumClasses));
    fn("UseRegularNms", (desc.m_UseRegularNms ? "true" : "false"));
    {
        std::stringstream ss;
        ss << "(" << desc.m_ScaleX <<  "," << desc.m_ScaleY << ")";
        fn("Scale(X,Y)", ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_ScaleW <<  "," << desc.m_ScaleH << ")";
        fn("Scale(W,H)", ss.str());
    }
}

void StringifyLayerParameters<DepthwiseConvolution2dDescriptor>::Serialize(
    ParameterStringifyFunction& fn,
    const DepthwiseConvolution2dDescriptor& desc)
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

    {
        std::stringstream ss;
        ss << "(" << desc.m_DilationX << "," << desc.m_DilationY << ")";
        fn("Dilation(X,Y)", ss.str());
    }

    fn("BiasEnabled",(desc.m_BiasEnabled ? "true" : "false"));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<ElementwiseUnaryDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                     const ElementwiseUnaryDescriptor& desc)
{
    fn("UnaryOperation", GetUnaryOperationAsCString(desc.m_Operation));
}

void StringifyLayerParameters<FakeQuantizationDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                     const FakeQuantizationDescriptor& desc)
{
    fn("Min", std::to_string(desc.m_Min));
    fn("Max", std::to_string(desc.m_Max));
}

void StringifyLayerParameters<FullyConnectedDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                   const FullyConnectedDescriptor& desc)
{
    fn("BiasEnabled", (desc.m_BiasEnabled ? "true" : "false"));
    fn("TransposeWeightMatrix", (desc.m_TransposeWeightMatrix ? "true" : "false"));
}

void StringifyLayerParameters<L2NormalizationDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                    const L2NormalizationDescriptor& desc)
{
    fn("Eps", std::to_string(desc.m_Eps));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<LstmDescriptor>::Serialize(ParameterStringifyFunction& fn, const LstmDescriptor& desc)
{
    fn("ActivationFunc", std::to_string(desc.m_ActivationFunc));
    fn("ClippingThresCell", std::to_string(desc.m_ClippingThresCell));
    fn("ClippingThresProj", std::to_string(desc.m_ClippingThresProj));
    fn("CifgEnabled", (desc.m_CifgEnabled ? "true" : "false"))   ;
    fn("PeepholeEnabled", (desc.m_PeepholeEnabled ? "true" : "false"))   ;
    fn("ProjectionEnabled", (desc.m_ProjectionEnabled ? "true" : "false"))   ;
    fn("LayerNormEnabled", (desc.m_LayerNormEnabled ? "true" : "false"));
}

void StringifyLayerParameters<MeanDescriptor>::Serialize(ParameterStringifyFunction& fn, const MeanDescriptor& desc)
{
    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_Axis)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("Axis", ss.str());
    }
    fn("KeepDims", (desc.m_KeepDims ? "true" : "false"));
}

void StringifyLayerParameters<NormalizationDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                  const NormalizationDescriptor& desc)
{
    fn("NormChannelType", GetNormalizationAlgorithmChannelAsCString(desc.m_NormChannelType));
    fn("NormMethodType", GetNormalizationAlgorithmMethodAsCString(desc.m_NormMethodType));
    fn("NormSize", std::to_string(desc.m_NormSize));
    fn("Alpha", std::to_string(desc.m_Alpha));
    fn("Beta", std::to_string(desc.m_Beta));
    fn("K", std::to_string(desc.m_K));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<OriginsDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                            const OriginsDescriptor& desc)
{
    fn("ConcatAxis", std::to_string(desc.GetConcatAxis()));

    uint32_t numViews = desc.GetNumViews();
    uint32_t numDims  = desc.GetNumDimensions();

    for (uint32_t view = 0; view < numViews; ++view)
    {
        std::stringstream key;
        key << "MergeTo#" << view;
        std::stringstream value;
        value << "[";
        auto viewData = desc.GetViewOrigin(view);

        for (uint32_t dim = 0; dim < numDims; ++dim)
        {
            if (dim > 0)
            {
                value << ",";
            }
            value << viewData[dim];
        }
        value << "]";
        fn(key.str(), value.str());
    }
}

void StringifyLayerParameters<PadDescriptor>::Serialize(ParameterStringifyFunction& fn, const PadDescriptor& desc)
{
    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_PadList)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << "[" << var.first << "," << var.second << "]";
            ++count;
        }
        fn("PadList", ss.str());
    }
    fn("PadValue", std::to_string(desc.m_PadValue));
    fn("PaddingMode", GetPaddingModeAsCString(desc.m_PaddingMode));
}

void StringifyLayerParameters<PreCompiledDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                const PreCompiledDescriptor& desc)
{
    fn("NumInputSlots", std::to_string(desc.m_NumInputSlots));
    fn("NumOutputSlots", std::to_string(desc.m_NumOutputSlots));
}

void StringifyLayerParameters<Pooling2dDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                              const Pooling2dDescriptor& desc)
{
    fn("Type", GetPoolingAlgorithmAsCString(desc.m_PoolType));
    {
        std::stringstream ss;
        ss << "(" << desc.m_PadTop    << "," << desc.m_PadLeft
           << "," << desc.m_PadBottom << "," << desc.m_PadRight << ")";
        fn("Padding(T,L,B,R)", ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_PoolWidth    << "," << desc.m_PoolHeight << ")";
        fn("(Width,Height)", ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_StrideX <<  "," << desc.m_StrideY << ")";
        fn("Stride(X,Y)", ss.str());
    }

    fn("OutputShapeRounding", GetOutputShapeRoundingAsCString(desc.m_OutputShapeRounding));
    fn("PaddingMethod", GetPaddingMethodAsCString(desc.m_PaddingMethod));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<Pooling3dDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                              const Pooling3dDescriptor& desc)
{
    fn("Type", GetPoolingAlgorithmAsCString(desc.m_PoolType));
    {
        std::stringstream ss;
        ss << "(" << desc.m_PadTop    << "," << desc.m_PadLeft
           << "," << desc.m_PadBottom << "," << desc.m_PadRight
           << "," << desc.m_PadFront  << "," << desc.m_PadBack << ")";
        fn("Padding(T,L,B,R,F,B)", ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_PoolWidth    << "," << desc.m_PoolHeight << "," << desc.m_PoolDepth << ")";
        fn("(Width,Height,Depth)", ss.str());
    }

    {
        std::stringstream ss;
        ss << "(" << desc.m_StrideX <<  "," << desc.m_StrideY << "," << desc.m_StrideZ << ")";
        fn("Stride(X,Y,Z)", ss.str());
    }

    fn("OutputShapeRounding", GetOutputShapeRoundingAsCString(desc.m_OutputShapeRounding));
    fn("PaddingMethod", GetPaddingMethodAsCString(desc.m_PaddingMethod));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<PermuteDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                            const PermuteDescriptor& desc)
{
    std::stringstream ss;
    ss <<  "[";
    bool addComma = false;
    for (auto it : desc.m_DimMappings)
    {
        if (addComma)
        {
            ss << ",";
        }
        ss << it;
        addComma = true;
    }
    ss << "]";

    fn("DimMappings",ss.str());
}

void StringifyLayerParameters<ReduceDescriptor>::Serialize(ParameterStringifyFunction& fn, const ReduceDescriptor& desc)
{
    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_vAxis)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("Axis", ss.str());
    }
    fn("KeepDims", (desc.m_KeepDims ? "true" : "false"));
    fn("ReduceOperation", GetReduceOperationAsCString(desc.m_ReduceOperation));
}

void StringifyLayerParameters<ReshapeDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                            const ReshapeDescriptor& desc)
{
    std::stringstream ss;
    ss << desc.m_TargetShape;
    fn("TargetShape",ss.str());
}

void StringifyLayerParameters<ResizeDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                           const ResizeDescriptor& desc)
{
    fn("TargetWidth", std::to_string(desc.m_TargetWidth));
    fn("TargetHeight", std::to_string(desc.m_TargetHeight));
    fn("ResizeMethod", GetResizeMethodAsCString(desc.m_Method));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
    fn("AlignCorners", std::to_string(desc.m_AlignCorners));
    fn("HalfPixelCenters", std::to_string(desc.m_HalfPixelCenters));
}

void StringifyLayerParameters<SoftmaxDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                            const SoftmaxDescriptor& desc)
{
    fn("Beta", std::to_string(desc.m_Beta));
    fn("Axis", std::to_string(desc.m_Axis));
}

void StringifyLayerParameters<SpaceToBatchNdDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                   const SpaceToBatchNdDescriptor& desc)
{
    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_BlockShape)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("BlockShape", ss.str());
    }

    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_PadList)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << "[" << var.first << "," << var.second << "]";
            ++count;
        }
        fn("PadList", ss.str());
    }

    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<SpaceToDepthDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                 const SpaceToDepthDescriptor& desc)
{
    fn("BlockSize", std::to_string(desc.m_BlockSize));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<StackDescriptor>::Serialize(ParameterStringifyFunction& fn, const StackDescriptor& desc)
{
    fn("Axis", std::to_string(desc.m_Axis));
    fn("NumInputs", std::to_string(desc.m_NumInputs));
    {
        std::stringstream ss;
        ss << desc.m_InputShape;
        fn("InputShape",ss.str());
    }
}

void StringifyLayerParameters<StridedSliceDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                                 const StridedSliceDescriptor& desc)
{
    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_Begin)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("Begin", ss.str());
    }

    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_End)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("End", ss.str());
    }

    {
        std::stringstream ss;
        int count = 0;
        for (auto&& var : desc.m_Stride)
        {
            if (count > 0)
            {
                ss << ",";
            }
            ss << var;
            ++count;
        }
        fn("Stride", ss.str());
    }

    fn("BeginMask", std::to_string(desc.m_BeginMask));
    fn("EndMask", std::to_string(desc.m_EndMask));
    fn("ShrinkAxisMask", std::to_string(desc.m_ShrinkAxisMask));
    fn("EllipsisMask", std::to_string(desc.m_EllipsisMask));
    fn("NewAxisMask", std::to_string(desc.m_NewAxisMask));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<TransposeConvolution2dDescriptor>::Serialize(
    ParameterStringifyFunction& fn,
    const TransposeConvolution2dDescriptor& desc)
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

    fn("BiasEnabled", (desc.m_BiasEnabled ? "true" : "false"));
    fn("DataLayout", GetDataLayoutName(desc.m_DataLayout));
}

void StringifyLayerParameters<TransposeDescriptor>::Serialize(ParameterStringifyFunction& fn,
                                                              const TransposeDescriptor& desc)
{
    std::stringstream ss;
    ss <<  "[";
    bool addComma = false;
    for (auto it : desc.m_DimMappings)
    {
        if (addComma)
        {
            ss << ",";
        }
        ss << it;
        addComma = true;
    }
    ss << "]";

    fn("DimMappings",ss.str());
}

void StringifyLayerParameters<ViewsDescriptor>::Serialize(ParameterStringifyFunction& fn, const ViewsDescriptor& desc)
{
    uint32_t numViews = desc.GetNumViews();
    uint32_t numDims  = desc.GetNumDimensions();
    for (uint32_t view = 0; view < numViews; ++view) {
        std::stringstream key;
        key << "ViewSizes#" << view;
        std::stringstream value;
        value << "[";
        auto viewData = desc.GetViewSizes(view);
        for (uint32_t dim = 0; dim < numDims; ++dim)
        {
            if (dim > 0)
            {
                value << ",";
            }
            value << viewData[dim];
        }
        value << "]";
        fn(key.str(), value.str());
    }
    StringifyLayerParameters<OriginsDescriptor>::Serialize(fn, desc.GetOrigins());
}

} // namespace armnn