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
struct FakeQuantizationDescriptor;
struct FullyConnectedDescriptor;
struct LstmDescriptor;
struct PermuteDescriptor;
struct NormalizationDescriptor;
struct L2NormalizationDescriptor;
struct MeanDescriptor;
struct PadDescriptor;
struct Pooling2dDescriptor;
struct ReshapeDescriptor;
struct SpaceToBatchNdDescriptor;
struct ResizeBilinearDescriptor;
struct SoftmaxDescriptor;
struct OriginsDescriptor;
struct ViewsDescriptor;
struct StridedSliceDescriptor;

using MergerDescriptor = OriginsDescriptor;
using SplitterDescriptor = ViewsDescriptor;
}
