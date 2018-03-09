//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

namespace armnn
{
struct ActivationDescriptor;
struct BatchNormalizationDescriptor;
struct Convolution2dDescriptor;
struct DepthwiseConvolution2dDescriptor;
struct FakeQuantizationDescriptor;
struct FullyConnectedDescriptor;
struct PermuteDescriptor;
struct NormalizationDescriptor;
struct Pooling2dDescriptor;
struct ReshapeDescriptor;
struct ResizeBilinearDescriptor;
struct SoftmaxDescriptor;
struct OriginsDescriptor;
struct ViewsDescriptor;

using MergerDescriptor = OriginsDescriptor;
using SplitterDescriptor = ViewsDescriptor;
}
