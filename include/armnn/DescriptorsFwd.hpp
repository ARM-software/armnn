//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{
struct ActivationDescriptor;
struct BatchNormalizationDescriptor;
struct BatchToSpaceNdDescriptor;
struct Convolution2dDescriptor;
struct DepthwiseConvolution2dDescriptor;
struct DetectionPostProcessDescriptor;
struct FakeQuantizationDescriptor;
struct FullyConnectedDescriptor;
struct L2NormalizationDescriptor;
struct LstmDescriptor;
struct MeanDescriptor;
struct NormalizationDescriptor;
struct OriginsDescriptor;
struct PadDescriptor;
struct PermuteDescriptor;
struct Pooling2dDescriptor;
struct PreCompiledDescriptor;
struct ReshapeDescriptor;
struct ResizeBilinearDescriptor;
struct ResizeDescriptor;
struct SoftmaxDescriptor;
struct SpaceToBatchNdDescriptor;
struct SpaceToDepthDescriptor;
struct StackDescriptor;
struct StridedSliceDescriptor;
struct TransposeConvolution2dDescriptor;
struct ViewsDescriptor;

// MergerDescriptor is deprecated use ConcatDescriptor instead
using MergerDescriptor = OriginsDescriptor;
using ConcatDescriptor = OriginsDescriptor;
using SplitterDescriptor = ViewsDescriptor;
}
