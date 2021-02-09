//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

class IStrategy
{
protected:
IStrategy() {}
virtual ~IStrategy() {}

public:
virtual void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) = 0;

virtual void FinishStrategy() {};

};


} // namespace armnn
