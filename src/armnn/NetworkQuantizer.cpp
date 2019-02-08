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
#include "NetworkQuantizerUtils.hpp"

#include "StaticRangeVisitor.hpp"
#include "QuantizerVisitor.hpp"
#include "OverrideInputRangeVisitor.hpp"

#include <vector>
#include <cmath>

namespace armnn
{

INetworkQuantizer* INetworkQuantizer::CreateRaw(INetwork* inputNetwork)
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

void NetworkQuantizer::OverrideInputRange(LayerBindingId layerId, float min, float max)
{
    const Graph& graph = boost::polymorphic_downcast<const Network*>(m_InputNetwork)->GetGraph();
    auto inputLayers = graph.GetInputLayers();

    // Walk the input layers of the graph and override the quantization parameters of the one with the given id
    OverrideInputRangeVisitor overrideInputRangeVisitor(m_Ranges, layerId, RangeTracker::MinMaxRange{min, max});
    VisitLayers(inputLayers, overrideInputRangeVisitor);
}

INetworkPtr NetworkQuantizer::ExportNetwork()
{
    const Graph& graph = boost::polymorphic_downcast<const Network*>(m_InputNetwork)->GetGraph().TopologicalSort();

    // Step 1) Walk the graph and register min/max values for intermediate tensors
    StaticRangeVisitor rangeVisitor(m_Ranges);
    VisitLayers(graph, rangeVisitor);

    // Step 2) Convert input InputNetwork to Quantized InputNetwork
    QuantizerVisitor quantizerVisitor(m_Ranges);
    VisitLayers(graph, quantizerVisitor);

    return quantizerVisitor.RetrieveFinalNetwork();
}

} //namespace armn
