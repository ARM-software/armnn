//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ILayerVisitor.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include "Graph.hpp"
#include "Layer.hpp"
#include "Network.hpp"
#include "NetworkQuantizer.hpp"

#include "StaticRangeVisitor.hpp"
#include "QuantizerVisitor.hpp"

#include <map>
#include <vector>
#include <cmath>

namespace armnn
{

INetworkQuantizer* INetworkQuantizer::CreateRaw(INetwork *inputNetwork)
{
    return new NetworkQuantizer(inputNetwork);
}

INetworkQuantizerPtr INetworkQuantizer::Create(INetwork* inputNetwork)
{
    return INetworkQuantizerPtr(CreateRaw(inputNetwork), &INetworkQuantizer::Destroy);
}

void INetworkQuantizer::Destroy(INetworkQuantizer *quantizer)
{
    delete boost::polymorphic_downcast<NetworkQuantizer*>(quantizer);
}

INetworkPtr NetworkQuantizer::ExportNetwork()
{
    const Graph& graph = boost::polymorphic_downcast<const Network*>(m_InputNetwork)->GetGraph().TopologicalSort();
    auto VisitLayers = [&graph](ILayerVisitor& visitor)
        {
            for (auto layer : graph)
            {
                layer->Accept(visitor);
            }
        };

    // Step 1) Walk the graph and register min/max values for intermediate tensors
    StaticRangeVisitor rangeVisitor;
    VisitLayers(rangeVisitor);

    // Step 2) Convert input InputNetwork to Quantized InputNetwork
    QuantizerVisitor quantizerVisitor(&rangeVisitor);
    VisitLayers(quantizerVisitor);

    return quantizerVisitor.RetrieveFinalNetwork();
}

} //namespace armn