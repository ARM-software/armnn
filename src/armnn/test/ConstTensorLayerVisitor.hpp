//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestLayerVisitor.hpp"
#include "LayersFwd.hpp"
#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <doctest/doctest.h>

namespace armnn
{

class TestConvolution2dLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestConvolution2dLayerVisitor(const Convolution2dDescriptor& convolution2dDescriptor,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(convolution2dDescriptor)
    {}

    virtual ~TestConvolution2dLayerVisitor() {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Convolution2d:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckDescriptor(static_cast<const armnn::Convolution2dDescriptor&>(descriptor));
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    void CheckDescriptor(const Convolution2dDescriptor& convolution2dDescriptor);

private:
    Convolution2dDescriptor m_Descriptor;
};

class TestDepthwiseConvolution2dLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestDepthwiseConvolution2dLayerVisitor(const DepthwiseConvolution2dDescriptor& descriptor,
                                                    const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(descriptor)
    {}

    virtual ~TestDepthwiseConvolution2dLayerVisitor() {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::DepthwiseConvolution2d:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckDescriptor(static_cast<const armnn::DepthwiseConvolution2dDescriptor&>(descriptor));
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    void CheckDescriptor(const DepthwiseConvolution2dDescriptor& convolution2dDescriptor);

private:
    DepthwiseConvolution2dDescriptor m_Descriptor;
};

class TestFullyConnectedLayerVistor : public TestLayerVisitor
{
public:
    explicit TestFullyConnectedLayerVistor(const FullyConnectedDescriptor& descriptor,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_Descriptor(descriptor)
    {}

    virtual ~TestFullyConnectedLayerVistor() {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::FullyConnected:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckDescriptor(static_cast<const armnn::FullyConnectedDescriptor&>(descriptor));
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    void CheckDescriptor(const FullyConnectedDescriptor& descriptor);
private:
    FullyConnectedDescriptor m_Descriptor;
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

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::BatchNormalization:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckDescriptor(static_cast<const armnn::BatchNormalizationDescriptor&>(descriptor));
                CheckConstTensors(m_Mean,     constants[0]);
                CheckConstTensors(m_Variance, constants[1]);
                CheckConstTensors(m_Beta,     constants[2]);
                CheckConstTensors(m_Gamma,    constants[3]);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
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

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Constant:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckConstTensors(m_Input, constants[0]);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

private:
    ConstTensor m_Input;
};

// Used to supply utility functions to the actual lstm test visitors
class LstmVisitor : public TestLayerVisitor
{
public:
    explicit LstmVisitor(const LstmInputParams& params,
                         const char* name = nullptr)
         : TestLayerVisitor(name)
         , m_InputParams(params) {}

protected:
    template<typename LayerType>
    void CheckInputParameters(const LayerType* layer, const LstmInputParams& inputParams);

    LstmInputParams m_InputParams;
};

template<typename LayerType>
void LstmVisitor::CheckInputParameters(const LayerType* layer, const LstmInputParams& inputParams)
{
    CheckConstTensorPtrs("OutputGateBias",
                         inputParams.m_OutputGateBias,
                         layer->m_BasicParameters.m_OutputGateBias);
    CheckConstTensorPtrs("InputToForgetWeights",
                         inputParams.m_InputToForgetWeights,
                         layer->m_BasicParameters.m_InputToForgetWeights);
    CheckConstTensorPtrs("InputToCellWeights",
                         inputParams.m_InputToCellWeights,
                         layer->m_BasicParameters.m_InputToCellWeights);
    CheckConstTensorPtrs("InputToOutputWeights",
                         inputParams.m_InputToOutputWeights,
                         layer->m_BasicParameters.m_InputToOutputWeights);
    CheckConstTensorPtrs("RecurrentToForgetWeights",
                         inputParams.m_RecurrentToForgetWeights,
                         layer->m_BasicParameters.m_RecurrentToForgetWeights);
    CheckConstTensorPtrs("RecurrentToCellWeights",
                         inputParams.m_RecurrentToCellWeights,
                         layer->m_BasicParameters.m_RecurrentToCellWeights);
    CheckConstTensorPtrs("RecurrentToOutputWeights",
                         inputParams.m_RecurrentToOutputWeights,
                         layer->m_BasicParameters.m_RecurrentToOutputWeights);
    CheckConstTensorPtrs("ForgetGateBias",
                         inputParams.m_ForgetGateBias,
                         layer->m_BasicParameters.m_ForgetGateBias);
    CheckConstTensorPtrs("CellBias",
                         inputParams.m_CellBias,
                         layer->m_BasicParameters.m_CellBias);

    CheckConstTensorPtrs("InputToInputWeights",
                         inputParams.m_InputToInputWeights,
                         layer->m_CifgParameters.m_InputToInputWeights);
    CheckConstTensorPtrs("RecurrentToInputWeights",
                         inputParams.m_RecurrentToInputWeights,
                         layer->m_CifgParameters.m_RecurrentToInputWeights);
    CheckConstTensorPtrs("InputGateBias",
                         inputParams.m_InputGateBias,
                         layer->m_CifgParameters.m_InputGateBias);

    CheckConstTensorPtrs("ProjectionBias",
                         inputParams.m_ProjectionBias,
                         layer->m_ProjectionParameters.m_ProjectionBias);
    CheckConstTensorPtrs("ProjectionWeights",
                         inputParams.m_ProjectionWeights,
                         layer->m_ProjectionParameters.m_ProjectionWeights);

    CheckConstTensorPtrs("CellToInputWeights",
                         inputParams.m_CellToInputWeights,
                         layer->m_PeepholeParameters.m_CellToInputWeights);
    CheckConstTensorPtrs("CellToForgetWeights",
                         inputParams.m_CellToForgetWeights,
                         layer->m_PeepholeParameters.m_CellToForgetWeights);
    CheckConstTensorPtrs("CellToOutputWeights",
                         inputParams.m_CellToOutputWeights,
                         layer->m_PeepholeParameters.m_CellToOutputWeights);

    CheckConstTensorPtrs("InputLayerNormWeights",
                         inputParams.m_InputLayerNormWeights,
                         layer->m_LayerNormParameters.m_InputLayerNormWeights);
    CheckConstTensorPtrs("ForgetLayerNormWeights",
                         inputParams.m_ForgetLayerNormWeights,
                         layer->m_LayerNormParameters.m_ForgetLayerNormWeights);
    CheckConstTensorPtrs("CellLayerNormWeights",
                         inputParams.m_CellLayerNormWeights,
                         layer->m_LayerNormParameters.m_CellLayerNormWeights);
    CheckConstTensorPtrs("OutputLayerNormWeights",
                         inputParams.m_OutputLayerNormWeights,
                         layer->m_LayerNormParameters.m_OutputLayerNormWeights);
}

class TestLstmLayerVisitor : public LstmVisitor
{
public:
    explicit TestLstmLayerVisitor(const LstmDescriptor& descriptor,
                                  const LstmInputParams& params,
                                  const char* name = nullptr)
        : LstmVisitor(params, name)
        , m_Descriptor(descriptor)
    {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Lstm:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckDescriptor(static_cast<const armnn::LstmDescriptor&>(descriptor));
                CheckInputParameters<const LstmLayer>(PolymorphicDowncast<const LstmLayer*>(layer), m_InputParams);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    void CheckDescriptor(const LstmDescriptor& descriptor);

private:
    LstmDescriptor m_Descriptor;
};

class TestQLstmLayerVisitor : public LstmVisitor
{
public:
    explicit TestQLstmLayerVisitor(const QLstmDescriptor& descriptor,
                                   const LstmInputParams& params,
                                   const char* name = nullptr)
            : LstmVisitor(params, name)
            , m_Descriptor(descriptor)
    {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::QLstm:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckDescriptor(static_cast<const armnn::QLstmDescriptor&>(descriptor));
                CheckInputParameters<const QLstmLayer>(PolymorphicDowncast<const QLstmLayer*>(layer), m_InputParams);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    void CheckDescriptor(const QLstmDescriptor& descriptor);

private:
    QLstmDescriptor m_Descriptor;
};


class TestQuantizedLstmLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestQuantizedLstmLayerVisitor(const QuantizedLstmInputParams& params,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_InputParams(params)
    {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::QuantizedLstm:
            {
                CheckLayerPointer(layer);
                CheckLayerName(name);
                CheckInputParameters(m_InputParams);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    void CheckInputParameters(const QuantizedLstmInputParams& params);

private:
    QuantizedLstmInputParams m_InputParams;
};


} // namespace armnn
