//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestLayerVisitor.hpp"
#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>

namespace armnn
{

class TestConvolution2dLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestConvolution2dLayerVisitor(const Convolution2dDescriptor& convolution2dDescriptor,
                                           const ConstTensor& weights,
                                           const Optional<ConstTensor>& biases,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(convolution2dDescriptor)
        , m_Weights(weights)
        , m_Biases(biases)
    {}

    virtual ~TestConvolution2dLayerVisitor() {}

    void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                 const Convolution2dDescriptor& convolution2dDescriptor,
                                 const ConstTensor& weights,
                                 const Optional<ConstTensor>& biases,
                                 const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckDescriptor(convolution2dDescriptor);
        CheckConstTensors(m_Weights, weights);
        CheckOptionalConstTensors(m_Biases, biases);
    }

protected:
    void CheckDescriptor(const Convolution2dDescriptor& convolution2dDescriptor);

private:
    Convolution2dDescriptor m_Descriptor;
    ConstTensor m_Weights;
    Optional<ConstTensor> m_Biases;
};

class TestDepthwiseConvolution2dLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestDepthwiseConvolution2dLayerVisitor(const DepthwiseConvolution2dDescriptor& descriptor,
                                                    const ConstTensor& weights,
                                                    const Optional<ConstTensor>& biases,
                                                    const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(descriptor)
        , m_Weights(weights)
        , m_Biases(biases)
    {}

    virtual ~TestDepthwiseConvolution2dLayerVisitor() {}

    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                          const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                          const ConstTensor& weights,
                                          const Optional<ConstTensor>& biases,
                                          const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckDescriptor(convolution2dDescriptor);
        CheckConstTensors(m_Weights, weights);
        CheckOptionalConstTensors(m_Biases, biases);
    }

protected:
    void CheckDescriptor(const DepthwiseConvolution2dDescriptor& convolution2dDescriptor);

private:
    DepthwiseConvolution2dDescriptor m_Descriptor;
    ConstTensor m_Weights;
    Optional<ConstTensor> m_Biases;
};

class TestFullyConnectedLayerVistor : public TestLayerVisitor
{
public:
    explicit TestFullyConnectedLayerVistor(const FullyConnectedDescriptor& descriptor,
                                           const ConstTensor& weights,
                                           const Optional<ConstTensor> biases,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(descriptor)
        , m_Weights(weights)
        , m_Biases(biases)
    {}

    virtual ~TestFullyConnectedLayerVistor() {}

    void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                  const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                  const ConstTensor& weights,
                                  const Optional<ConstTensor>& biases,
                                  const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckDescriptor(fullyConnectedDescriptor);
        CheckConstTensors(m_Weights, weights);
        CheckOptionalConstTensors(m_Biases, biases);
    }

protected:
    void CheckDescriptor(const FullyConnectedDescriptor& descriptor);
private:
    FullyConnectedDescriptor m_Descriptor;
    ConstTensor m_Weights;
    Optional<ConstTensor> m_Biases;
};

class TestBatchNormalizationLayerVisitor : public TestLayerVisitor
{
public:
    TestBatchNormalizationLayerVisitor(const BatchNormalizationDescriptor& descriptor,
                                       const ConstTensor& mean,
                                       const ConstTensor& variance,
                                       const ConstTensor& beta,
                                       const ConstTensor& gamma,
                                       const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(descriptor)
        , m_Mean(mean)
        , m_Variance(variance)
        , m_Beta(beta)
        , m_Gamma(gamma)
    {}

    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& descriptor,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckDescriptor(descriptor);
        CheckConstTensors(m_Mean, mean);
        CheckConstTensors(m_Variance, variance);
        CheckConstTensors(m_Beta, beta);
        CheckConstTensors(m_Gamma, gamma);
    }

protected:
    void CheckDescriptor(const BatchNormalizationDescriptor& descriptor);

private:
    BatchNormalizationDescriptor m_Descriptor;
    ConstTensor m_Mean;
    ConstTensor m_Variance;
    ConstTensor m_Beta;
    ConstTensor m_Gamma;
};

class TestConstantLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestConstantLayerVisitor(const ConstTensor& input,
                                      const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Input(input)
    {}

    void VisitConstantLayer(const IConnectableLayer* layer,
                            const ConstTensor& input,
                            const char* name = nullptr)
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckConstTensors(m_Input, input);
    }

private:
    ConstTensor m_Input;
};

class TestLstmLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestLstmLayerVisitor(const LstmDescriptor& descriptor,
                                  const LstmInputParams& params,
                                  const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(descriptor)
        , m_InputParams(params)
    {}

    void VisitLstmLayer(const IConnectableLayer* layer,
                        const LstmDescriptor& descriptor,
                        const LstmInputParams& params,
                        const char* name = nullptr)
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckDescriptor(descriptor);
        CheckInputParameters(params);
    }

protected:
    void CheckDescriptor(const LstmDescriptor& descriptor);
    void CheckInputParameters(const LstmInputParams& inputParams);
    void CheckConstTensorPtrs(const std::string& name, const ConstTensor* expected, const ConstTensor* actual);

private:
    LstmDescriptor m_Descriptor;
    LstmInputParams m_InputParams;
};

class TestQLstmLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestQLstmLayerVisitor(const QLstmDescriptor& descriptor,
                                   const LstmInputParams& params,
                                   const char* name = nullptr)
            : TestLayerVisitor(name)
            , m_Descriptor(descriptor)
            , m_InputParams(params)
    {}

    void VisitQLstmLayer(const IConnectableLayer* layer,
                         const QLstmDescriptor& descriptor,
                         const LstmInputParams& params,
                         const char* name = nullptr)
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckDescriptor(descriptor);
        CheckInputParameters(params);
    }

protected:
    void CheckDescriptor(const QLstmDescriptor& descriptor);
    void CheckInputParameters(const LstmInputParams& inputParams);
    void CheckConstTensorPtrs(const std::string& name, const ConstTensor* expected, const ConstTensor* actual);

private:
    QLstmDescriptor m_Descriptor;
    LstmInputParams m_InputParams;
};


class TestQuantizedLstmLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestQuantizedLstmLayerVisitor(const QuantizedLstmInputParams& params,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_InputParams(params)
    {}

    void VisitQuantizedLstmLayer(const IConnectableLayer* layer,
                                 const QuantizedLstmInputParams& params,
                                 const char* name = nullptr)
    {
        CheckLayerPointer(layer);
        CheckLayerName(name);
        CheckInputParameters(params);
    }

protected:
    void CheckInputParameters(const QuantizedLstmInputParams& inputParams);
    void CheckConstTensorPtrs(const std::string& name, const ConstTensor* expected, const ConstTensor* actual);

private:
    QuantizedLstmInputParams m_InputParams;
};


} // namespace armnn
