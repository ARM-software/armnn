//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <functional>
#include <armnn/Descriptors.hpp>

namespace armnn
{

using ParameterStringifyFunction = std::function<void(const std::string& name, const std::string& value)>;

///
/// StringifyLayerParameters allows serializing layer parameters to string.
/// The default implementation is a no-op because this operation is considered
/// non-vital for ArmNN and thus we allow adding new layer parameters without
/// supplying the corresponding stringify functionality.
///
template <typename LayerParameter>
struct StringifyLayerParameters
{
    static void Serialize(ParameterStringifyFunction&, const LayerParameter&) {}
};

template <> struct StringifyLayerParameters<ActivationDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ActivationDescriptor& desc);
};

template <> struct StringifyLayerParameters<BatchNormalizationDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const BatchNormalizationDescriptor& desc);
};

template <> struct StringifyLayerParameters<BatchToSpaceNdDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const BatchToSpaceNdDescriptor& desc);
};

template <> struct StringifyLayerParameters<ChannelShuffleDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ChannelShuffleDescriptor& desc);
};

template <> struct StringifyLayerParameters<ComparisonDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ComparisonDescriptor& desc);
};

template <> struct StringifyLayerParameters<Convolution2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const Convolution2dDescriptor& desc);
};

template <> struct StringifyLayerParameters<Convolution3dDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const Convolution3dDescriptor& desc);
};

template <> struct StringifyLayerParameters<DetectionPostProcessDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const DetectionPostProcessDescriptor& desc);
};

template <> struct StringifyLayerParameters<DepthwiseConvolution2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const DepthwiseConvolution2dDescriptor& desc);
};

template <> struct StringifyLayerParameters<ElementwiseUnaryDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ElementwiseUnaryDescriptor& desc);
};

template <> struct StringifyLayerParameters<FakeQuantizationDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const FakeQuantizationDescriptor& desc);
};

template <> struct StringifyLayerParameters<FullyConnectedDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const FullyConnectedDescriptor& desc);
};

template <> struct StringifyLayerParameters<L2NormalizationDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const L2NormalizationDescriptor& desc);
};

template <> struct StringifyLayerParameters<LstmDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const LstmDescriptor& desc);
};

template <> struct StringifyLayerParameters<MeanDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const MeanDescriptor& desc);
};

template <> struct StringifyLayerParameters<NormalizationDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const NormalizationDescriptor& desc);
};

template <> struct StringifyLayerParameters<OriginsDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const OriginsDescriptor& desc);
};

template <> struct StringifyLayerParameters<PadDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const PadDescriptor& desc);
};
template <> struct StringifyLayerParameters<PermuteDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const PermuteDescriptor& desc);
};

template <> struct StringifyLayerParameters<Pooling2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const Pooling2dDescriptor& desc);
};

template <> struct StringifyLayerParameters<Pooling3dDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const Pooling3dDescriptor& desc);
};

template <> struct StringifyLayerParameters<PreCompiledDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const PreCompiledDescriptor& desc);
};

template <> struct StringifyLayerParameters<ReduceDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ReduceDescriptor& desc);
};

template <> struct StringifyLayerParameters<ReshapeDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ReshapeDescriptor& desc);
};

template <> struct StringifyLayerParameters<ResizeDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ResizeDescriptor& desc);
};

template <> struct StringifyLayerParameters<SpaceToBatchNdDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const SpaceToBatchNdDescriptor& desc);
};

template <> struct StringifyLayerParameters<SpaceToDepthDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const SpaceToDepthDescriptor& desc);
};

template <> struct StringifyLayerParameters<StackDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const StackDescriptor& desc);
};

template <> struct StringifyLayerParameters<StridedSliceDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const StridedSliceDescriptor& desc);
};

template <> struct StringifyLayerParameters<SoftmaxDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const SoftmaxDescriptor& desc);
};

template <> struct StringifyLayerParameters<TransposeConvolution2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const TransposeConvolution2dDescriptor& desc);
};

template <> struct StringifyLayerParameters<TransposeDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const TransposeDescriptor& desc);
};

template <> struct StringifyLayerParameters<ViewsDescriptor>
{
    static void Serialize(ParameterStringifyFunction& fn, const ViewsDescriptor& desc);
};

} // namespace armnn