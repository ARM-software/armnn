//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StaticRangeVisitor.hpp"

#include <boost/core/ignore_unused.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

StaticRangeVisitor::StaticRangeVisitor(RangeTracker& rangeTracker)
    : m_RangeTracker(rangeTracker)
{}

void StaticRangeVisitor::SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max)
{
    m_RangeTracker.SetRange(layer, outputIdx, min, max);
}

void StaticRangeVisitor::VisitAdditionLayer(const IConnectableLayer* layer, const char* name)
{
    SetRange(layer, 0, -20.f, 20.f);
}

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

void StaticRangeVisitor::VisitConvolution2dLayer(const IConnectableLayer* layer,
                                                 const Convolution2dDescriptor& convolution2dDescriptor,
                                                 const ConstTensor& weights,
                                                 const Optional<ConstTensor>& biases,
                                                 const char* name)
{
    boost::ignore_unused(convolution2dDescriptor);
    boost::ignore_unused(weights);
    boost::ignore_unused(biases);
    boost::ignore_unused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
}

void StaticRangeVisitor::VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                                          const DepthwiseConvolution2dDescriptor& desc,
                                                          const ConstTensor& weights,
                                                          const Optional<ConstTensor>& biases,
                                                          const char* name)
{
    boost::ignore_unused(desc);
    boost::ignore_unused(weights);
    boost::ignore_unused(biases);
    boost::ignore_unused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
}

void StaticRangeVisitor::VisitActivationLayer(const IConnectableLayer* layer,
                                              const ActivationDescriptor& activationDescriptor,
                                              const char* name)
{
    switch (activationDescriptor.m_Function)
    {
        // Range is 0, 15 for Abs, Linear, ReLu and Soft ReLu
        case ActivationFunction::Abs:
        case ActivationFunction::Linear:
        case ActivationFunction::ReLu:
        case ActivationFunction::SoftReLu:
            SetRange(layer, 0, 0.f, 15.f);
            break;
        case ActivationFunction::BoundedReLu:
            SetRange(layer, 0, 0.f, activationDescriptor.m_A);
            break;
        case ActivationFunction::TanH:
            SetRange(layer, 0, -1.f, 1.f);
            break;
        case ActivationFunction::LeakyReLu:
            SetRange(layer, 0, -5.f, 15.f);
            break;
        default:
            SetRange(layer, 0, -15.f, 15.f);
            break;
    }
}

void StaticRangeVisitor::VisitFullyConnectedLayer(const IConnectableLayer *layer,
                                                  const FullyConnectedDescriptor& desc,
                                                  const ConstTensor& weights,
                                                  const Optional<ConstTensor>& biases,
                                                  const char *name)
{
    boost::ignore_unused(desc);
    boost::ignore_unused(weights);
    boost::ignore_unused(biases);
    boost::ignore_unused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
}

void StaticRangeVisitor::VisitSoftmaxLayer(const IConnectableLayer* layer,
                                           const SoftmaxDescriptor& softmaxDescriptor,
                                           const char* name)
{
    boost::ignore_unused(softmaxDescriptor);
    SetRange(layer, 0, 0.f, 1.f);
}

} //namespace armnn
