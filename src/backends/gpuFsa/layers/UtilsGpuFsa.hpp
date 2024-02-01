//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Descriptors.hpp"
#include "arm_compute/dynamic_fusion/sketch/attributes/Conv2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/DepthwiseConv2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"

/// Utility function used to setup an arm_compute::Conv2dAttributes object from given descriptor
/// @param[in] armnn::Convolution2dDescriptor
/// @return arm_compute::experimental::dynamic_fusion::Conv2dAttributes
arm_compute::experimental::dynamic_fusion::Conv2dAttributes
CreateConv2dAttributes(const armnn::Convolution2dDescriptor& descriptor);

/// Utility function used to setup an arm_compute::DepthwiseConv2dAttributes object from given descriptor
/// @param[in] armnn::DepthwiseConvolution2dDescriptor
/// @return arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes
arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes
CreateDWConv2dAttributes(const armnn::DepthwiseConvolution2dDescriptor& descriptor, 
                         const unsigned int aclDepthMultiplier);

/// Utility function used to setup an arm_compute::Pool2dAttributes object from given descriptor
/// @param[in] armnn::Pooling2dDescriptor
/// @return arm_compute::experimental::dynamic_fusion::Pool2dAttributes
arm_compute::experimental::dynamic_fusion::Pool2dAttributes
CreatePool2dAttributes(const armnn::Pooling2dDescriptor& descriptor);
