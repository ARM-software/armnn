//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

class NetworkQuantizer : public INetworkQuantizer
{
public:
    NetworkQuantizer(INetwork* inputNetwork) : m_InputNetwork(inputNetwork) {}

    INetworkPtr ExportNetwork() override;

private:
    INetwork* m_InputNetwork;
};

} //namespace armnn