//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StaticRangeVisitor.hpp"

#include <boost/core/ignore_unused.hpp>

namespace armnn
{

void StaticRangeVisitor::SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max)
{
    auto& ranges = m_GuidToRangesMap[layer->GetGuid()];

    if (ranges.size() < layer->GetNumOutputSlots())
    {
        ranges.resize(layer->GetNumOutputSlots());
    }
    ranges[outputIdx] = std::make_pair(min, max);
}

StaticRangeVisitor::MinMaxRange StaticRangeVisitor::GetRange(LayerGuid guid, unsigned int idx) const
{
    auto found = m_GuidToRangesMap.find(guid);
    if (found != m_GuidToRangesMap.end())
    {
        return found->second.at(idx);
    }
    return DefaultRange();
}

void StaticRangeVisitor::VisitAdditionLayer(const IConnectableLayer *layer, const char *name)
{
    SetRange(layer, 0, -20.f, 20.f);
};

void StaticRangeVisitor::VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                                      const BatchNormalizationDescriptor& desc,
                                                      const ConstTensor& mean,
                                                      const ConstTensor& variance,
                                                      const ConstTensor& beta,
                                                      const ConstTensor& gamma,
                                                      const char* name)
{
    boost::ignore_unused(desc);
    boost::ignore_unused(mean);
    boost::ignore_unused(variance);
    boost::ignore_unused(beta);
    boost::ignore_unused(gamma);
    boost::ignore_unused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
}

} //namespace armnn