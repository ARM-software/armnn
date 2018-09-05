//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

namespace armnn
{

struct QueueDescriptor;
template <typename LayerDescriptor>
struct QueueDescriptorWithParameters;
struct SoftmaxQueueDescriptor;
struct SplitterQueueDescriptor;
struct MergerQueueDescriptor;
struct ActivationQueueDescriptor;
struct FullyConnectedQueueDescriptor;
struct PermuteQueueDescriptor;
struct Pooling2dQueueDescriptor;
struct Convolution2dQueueDescriptor;
struct NormalizationQueueDescriptor;
struct MultiplicationQueueDescriptor;
struct BatchNormalizationQueueDescriptor;
struct FakeQuantizationQueueDescriptor;
struct ReshapeQueueDescriptor;

} // namespace armnn