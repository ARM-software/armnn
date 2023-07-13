//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{
struct BaseDescriptor;

struct ActivationDescriptor;
struct ArgMinMaxDescriptor;
struct BatchMatMulDescriptor;
struct BatchNormalizationDescriptor;
struct BatchToSpaceNdDescriptor;
struct ChannelShuffleDescriptor;
struct ComparisonDescriptor;
struct Convolution2dDescriptor;
struct Convolution3dDescriptor;
struct DepthwiseConvolution2dDescriptor;
struct DetectionPostProcessDescriptor;
struct ElementwiseBinaryDescriptor;
struct ElementwiseUnaryDescriptor;
struct FakeQuantizationDescriptor;
struct FillDescriptor;
struct FullyConnectedDescriptor;
struct GatherDescriptor;
struct InstanceNormalizationDescriptor;
struct L2NormalizationDescriptor;
struct LogicalBinaryDescriptor;
struct LstmDescriptor;
struct MeanDescriptor;
struct NormalizationDescriptor;
struct OriginsDescriptor;
struct PadDescriptor;
struct PermuteDescriptor;
struct Pooling2dDescriptor;
struct Pooling3dDescriptor;
struct PreCompiledDescriptor;
struct QLstmDescriptor;
struct ReshapeDescriptor;
struct ResizeDescriptor;
struct ReduceDescriptor;
struct SliceDescriptor;
struct SoftmaxDescriptor;
struct SpaceToBatchNdDescriptor;
struct SpaceToDepthDescriptor;
struct StackDescriptor;
struct StandInDescriptor;
struct StridedSliceDescriptor;
struct TransposeConvolution2dDescriptor;
struct TransposeDescriptor;
struct ViewsDescriptor;

using ConcatDescriptor       = OriginsDescriptor;
using DepthToSpaceDescriptor = SpaceToDepthDescriptor;
using LogSoftmaxDescriptor   = SoftmaxDescriptor;
/// MergerDescriptor is deprecated, use ConcatDescriptor instead
using MergerDescriptor       = OriginsDescriptor;
using SplitterDescriptor     = ViewsDescriptor;
using UnidirectionalSequenceLstmDescriptor = LstmDescriptor;

} // namespace armnn
