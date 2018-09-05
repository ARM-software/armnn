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

using ParameterStringifyFunction = std::function<void(const std::string & name, const std::string & value)>;

///
/// StringifyLayerParameters allows serializing layer parameters to string.
/// The default implementation is a no-op because this operation is considered
/// non-vital for ArmNN and thus we allow adding new layer parameters without
/// supplying the corresponding stringify functionality.
///
template <typename LayerParameter>
struct StringifyLayerParameters
{
    static void Serialize(ParameterStringifyFunction &, const LayerParameter &) {}
};

template <> struct StringifyLayerParameters<PermuteDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const PermuteDescriptor & desc);
};

template <> struct StringifyLayerParameters<ReshapeDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const ReshapeDescriptor & desc);
};

template <> struct StringifyLayerParameters<ActivationDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const ActivationDescriptor & desc);
};

template <> struct StringifyLayerParameters<Convolution2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const Convolution2dDescriptor & desc);
};

template <> struct StringifyLayerParameters<BatchNormalizationDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const BatchNormalizationDescriptor & desc);
};

template <> struct StringifyLayerParameters<DepthwiseConvolution2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const DepthwiseConvolution2dDescriptor & desc);
};

template <> struct StringifyLayerParameters<Pooling2dDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const Pooling2dDescriptor & desc);
};

template <> struct StringifyLayerParameters<SoftmaxDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const SoftmaxDescriptor & desc);
};

template <> struct StringifyLayerParameters<FullyConnectedDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const FullyConnectedDescriptor & desc);
};

template <> struct StringifyLayerParameters<OriginsDescriptor>
{
    static void Serialize(ParameterStringifyFunction & fn, const OriginsDescriptor & desc);
};

}