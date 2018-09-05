//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Layer.hpp"

namespace armnn
{

enum class GraphEvent
{
    LayerAdded,
    LayerErased
};

class IGraphObservable
{
public:
    virtual void Update(Layer* graphLayer) = 0;

protected:
    virtual ~IGraphObservable() = default;
};

} //namespace armnn

