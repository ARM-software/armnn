//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"

#include <Network.hpp>
#include <ResolveType.hpp>
#include <armnn/INetwork.hpp>
#include <test/TestUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <QuantizeHelper.hpp>
#include <string>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(Optimizer)

namespace
{
const float   g_qScale  = 1.0f;
const int32_t g_qOffset = 0;

template<typename T>
std::vector<T> GetVector(unsigned int size, float initial, float increment)
{
    std::vector<float> typeVector(size, initial);
    std::vector<T>     vector(size);

    if (size > 1)
    {
        for (unsigned int i = 0; i < size; ++i)
        {
            vector[i] = T(initial + (increment * static_cast<float>(i)));
        }
    }
    return vector;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct Convolution2dTest
{
    using LayerType = armnn::Convolution2dLayer;
    static std::string GetReceiverLayerName() { return "Convolution2d"; };
    static const bool isElementWise = false;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }  // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 3, 3, 4}); }  // NHWCout
    static TensorShape GetWeightsShape() { return TensorShape( {4, 2, 2, 3}); }  // CoutHWCin

    constexpr static const unsigned int inputSize  = 48; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 36; // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        Convolution2dDescriptor descriptor;
        descriptor.m_BiasEnabled = false;
        descriptor.m_DataLayout  = DataLayout::NHWC;
        descriptor.m_StrideX     = 1;
        descriptor.m_StrideY     = 1;

        std::vector<float> weightsData   = {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
        std::vector<T>     weightsVector = armnnUtils::QuantizedVector<T>(weightsData, g_qScale, g_qOffset);
        TensorInfo         weightsInfo(GetWeightsShape(), ArmnnType, g_qScale, g_qOffset);
        ConstTensor        weights(weightsInfo, weightsVector);
        Optional<ConstTensor> optionalBias;

        return network->AddConvolution2dLayer(descriptor, weights, optionalBias, name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct DepthwiseConvolution2dTest
{
public:
    using LayerType = armnn::DepthwiseConvolution2dLayer;
    static std::string GetReceiverLayerName() { return "DepthwiseConvolution2d"; };
    static const bool isElementWise = false;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }   // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 3, 3, 12}); }  // NHWCout
    static TensorShape GetWeightsShape() { return TensorShape( {4, 3, 2, 2}); }   // MCinHW

    constexpr static const unsigned int inputSize  = 48; //batchIn * heightIn * widthIn * channelIn;
    constexpr static const unsigned int outputSize = 108; //batchOut * heightOut * widthOut * channelOut;

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        DepthwiseConvolution2dDescriptor descriptor;
        descriptor.m_BiasEnabled = false;
        descriptor.m_DataLayout  = DataLayout::NHWC;
        descriptor.m_StrideX     = 1;
        descriptor.m_StrideY     = 1;

        std::vector<float> weightsData   = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
        std::vector<T>     weightsVector = armnnUtils::QuantizedVector<T>(weightsData, g_qScale, g_qOffset);
        TensorInfo         weightsInfo(GetWeightsShape(), ArmnnType, g_qScale, g_qOffset);
        ConstTensor        weights(weightsInfo, weightsVector);
        Optional<ConstTensor> optionalBias;

        return network->AddDepthwiseConvolution2dLayer(descriptor, weights, optionalBias, name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct FullyConnectedTest
{
public:
    using LayerType = armnn::FullyConnectedLayer;
    static std::string GetReceiverLayerName() { return "FullyConnected"; };
    static const bool isElementWise = false;

    static TensorShape GetInputShape()   { return TensorShape( {2, 5, 1, 1}); } // NCinHW
    static TensorShape GetOutputShape()  { return TensorShape( {2, 3}); }       // NCout
    static TensorShape GetWeightsShape() { return TensorShape( {5, 3}); }       // CinCout

    constexpr static const unsigned int inputSize  = 10; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 6;  // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        FullyConnectedDescriptor descriptor;
        descriptor.m_BiasEnabled = false;

        std::vector<float> weightsData   = { 1,  2,  3,  4,  5,
                                             6,  7,  8,  9, 10,
                                            11, 12, 13, 14, 15};
        std::vector<T>     weightsVector = armnnUtils::QuantizedVector<T>(weightsData, g_qScale, g_qOffset);
        TensorInfo         weightsInfo(GetWeightsShape(), ArmnnType, g_qScale, g_qOffset);
        ConstTensor        weights(weightsInfo, weightsVector);
        Optional<ConstTensor> optionalBias;

        return network->AddFullyConnectedLayer(descriptor, weights, optionalBias, name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct BatchNormTest
{
public:
    using LayerType = armnn::BatchNormalizationLayer;
    static std::string GetReceiverLayerName() { return "BatchNorm"; };
    static const bool isElementWise = false;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }  // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 4, 4, 3}); }  // NHWCout

    constexpr static const unsigned int inputSize  = 48; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 48; // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        BatchNormalizationDescriptor descriptor;
        descriptor.m_DataLayout = DataLayout::NHWC;

        std::vector<T> betaVector     = GetVector<T>(GetOutputShape()[3], 0.0f, 0.2f);
        std::vector<T> gammaVector    = GetVector<T>(GetOutputShape()[3], 0.5f, 0.1f);
        std::vector<T> meanVector     = GetVector<T>(GetOutputShape()[3], 0.1f, 0.1f);
        std::vector<T> varianceVector = GetVector<T>(GetOutputShape()[3], 1.0f, 0.1f);

        const unsigned int outputChannelSize[] = { GetOutputShape()[3] };
        ConstTensor beta(TensorInfo(1, outputChannelSize, ArmnnType), betaVector);
        ConstTensor gamma(TensorInfo(1, outputChannelSize, ArmnnType), gammaVector);
        ConstTensor mean(TensorInfo(1, outputChannelSize, ArmnnType), meanVector);
        ConstTensor variance(TensorInfo(1, outputChannelSize, ArmnnType), varianceVector);

        return network->AddBatchNormalizationLayer(descriptor, mean, variance, beta, gamma, name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct MultiplicationTest
{
    using LayerType = armnn::MultiplicationLayer;
    static std::string GetReceiverLayerName() { return "Multiplication"; };
    static const bool isElementWise = true;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }  // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 4, 4, 3}); }  // NHWCout

    constexpr static const unsigned int inputSize  = 48; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 48; // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        return network->AddMultiplicationLayer(name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct AdditionTest
{
    using LayerType = armnn::AdditionLayer;
    static std::string GetReceiverLayerName() { return "Addition"; };
    static const bool isElementWise = true;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }  // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 4, 4, 3}); }  // NHWCout

    constexpr static const unsigned int inputSize  = 48; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 48; // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        return network->AddAdditionLayer(name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct SubtractionTest
{
    using LayerType = armnn::SubtractionLayer;
    static std::string GetReceiverLayerName() { return "Subtraction"; };
    static const bool isElementWise = true;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }  // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 4, 4, 3}); }  // NHWCout

    constexpr static const unsigned int inputSize  = 48; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 48; // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        return network->AddSubtractionLayer(name);
    }
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct DivisionTest
{
    using LayerType = armnn::DivisionLayer;
    static std::string GetReceiverLayerName() { return "Division"; };
    static const bool isElementWise = true;

    static TensorShape GetInputShape()   { return TensorShape( {1, 4, 4, 3}); }  // NHWCin
    static TensorShape GetOutputShape()  { return TensorShape( {1, 4, 4, 3}); }  // NHWCout

    constexpr static const unsigned int inputSize  = 48; // batchIn * heightIn * widthIn * channelIn
    constexpr static const unsigned int outputSize = 48; // batchOut * heightOut * widthOut * channelOut

    static IConnectableLayer* AddReceiverLayer(INetwork* network,
                                               const char* name)
    {
        return network->AddDivisionLayer(name);
    }
};

} // namespace

template<typename LayerTest,
         armnn::DataType ArmnnType>
INetworkPtr CreatNetwork(ActivationDescriptor activationDescriptor, bool preventFusing)
{
    // Create a network
    INetworkPtr network = INetwork::Create();

    IConnectableLayer* inputLayer = network->AddInputLayer(0);

    IConnectableLayer* receiverLayer = LayerTest::AddReceiverLayer(network.get(),
                                                                   "receiverLayer");

    IConnectableLayer* activationLayer = network->AddActivationLayer(activationDescriptor,
                                                                     "activation");

    IConnectableLayer* outputLayer  = network->AddOutputLayer(0);
    IConnectableLayer* output2Layer = preventFusing?network->AddOutputLayer(1):nullptr;

    // Define layers information
    TensorInfo inputInfo(LayerTest::GetInputShape(), ArmnnType, g_qScale, g_qOffset);
    TensorInfo outputInfo(LayerTest::GetOutputShape(), ArmnnType, g_qScale, g_qOffset);

    // Set layer information
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    receiverLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    activationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Connect layers
    inputLayer->GetOutputSlot(0).Connect(receiverLayer->GetInputSlot(0));
    receiverLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    if (LayerTest::isElementWise)
    {
        inputLayer->GetOutputSlot(0).Connect(receiverLayer->GetInputSlot(1));
    }
    if (preventFusing)
    {
        receiverLayer->GetOutputSlot(0).Connect(output2Layer->GetInputSlot(0));
    }

    return network;
}

template<typename LayerTest,
         armnn::DataType ArmnnType,
         typename LayerType = typename LayerTest::LayerType,
         typename T = armnn::ResolveType<ArmnnType>>
void FuseActivationIntoPreviousLayerTest(ActivationDescriptor activationDescriptor, float tolerance, armnn::Compute
backendId)
{
    // FIRST NETWORK: Fused
    // Construct ArmNN network
    INetworkPtr networkFused = CreatNetwork<LayerTest, ArmnnType>(activationDescriptor, false);

    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions()); // default options

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNetFused = Optimize(*networkFused, {backendId}, run->GetDeviceSpec());

    Graph graphFused = PolymorphicDowncast<OptimizedNetwork*>(optNetFused.get())->GetGraph();

    auto checkFusedConv2d = [](const armnn::Layer* const layer)->bool {
        return IsLayerOfType<LayerType>(layer) &&
            (layer->GetNameStr() == "fused-activation-into-receiverLayer");
    };

    BOOST_CHECK_MESSAGE(3 == graphFused.GetNumLayers(), LayerTest::GetReceiverLayerName());
    BOOST_TEST(CheckSequence(graphFused.cbegin(),
                             graphFused.cend(),
                             &IsLayerOfType<InputLayer>,
                             checkFusedConv2d,
                             &IsLayerOfType<OutputLayer>));

    // Load network into runtime
    NetworkId networkIdentifier;
    BOOST_TEST(run->LoadNetwork(networkIdentifier, std::move(optNetFused)) == Status::Success);

    //Creates structures for inputs and outputs.
    std::vector<float> data = GetVector<float>(LayerTest::inputSize, 1.0f, 0.1f);
    std::vector<T> inputDataFused = armnnUtils::QuantizedVector<T>(data, g_qScale, g_qOffset);
    std::vector<T> outputDataFused(LayerTest::outputSize);

    InputTensors  inputTensorsFused{
        {0, ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0), inputDataFused.data())}};
    OutputTensors outputTensorsFused{
        {0, Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputDataFused.data())}};

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensorsFused, outputTensorsFused);

    // SECOND NETWORK: NotFused
    // Construct ArmNN network
    INetworkPtr networkNotFused = CreatNetwork<LayerTest, ArmnnType>(activationDescriptor, true);

    // Create ArmNN runtime
    IRuntimePtr runNotFused = IRuntime::Create(IRuntime::CreationOptions()); // default options

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNetNotFused = Optimize(*networkNotFused, {backendId}, runNotFused->GetDeviceSpec());

    Graph graphNotFused = PolymorphicDowncast<OptimizedNetwork*>(optNetNotFused.get())->GetGraph();

    BOOST_CHECK(5 == graphNotFused.GetNumLayers());
    BOOST_TEST(CheckSequence(graphNotFused.cbegin(),
                             graphNotFused.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<LayerType>,
                             &IsLayerOfType<armnn::ActivationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    // Load network into runtime
    NetworkId networkIdentifierNotFused;
    BOOST_TEST(runNotFused->LoadNetwork(networkIdentifierNotFused, std::move(optNetNotFused)) == Status::Success);

    //Creates structures for inputs and outputs.
    std::vector<T> inputDataNotFused = armnnUtils::QuantizedVector<T>(data, g_qScale, g_qOffset);
    std::vector<T> outputDataNotFused(LayerTest::outputSize);
    std::vector<T> outputData2NotFused(LayerTest::outputSize);

    InputTensors  inputTensorsNotFused{
        {0, ConstTensor(runNotFused->GetInputTensorInfo(networkIdentifierNotFused, 0), inputDataNotFused.data())}};
    OutputTensors outputTensorsNotFused{
        {0, Tensor(runNotFused->GetOutputTensorInfo(networkIdentifierNotFused, 0), outputDataNotFused.data())},
        {1, Tensor(runNotFused->GetOutputTensorInfo(networkIdentifierNotFused, 1), outputData2NotFused.data())}};

    // Execute network
    runNotFused->EnqueueWorkload(networkIdentifierNotFused, inputTensorsNotFused, outputTensorsNotFused);

    // Check the output of the fused-activation matches with the output of the activation in the "NotFused" network
    for (unsigned int n = 0; n < outputDataFused.size(); ++n)
    {
        BOOST_CHECK_CLOSE(static_cast<float>(outputDataFused[n]), static_cast<float>(outputDataNotFused[n]),
                          T(tolerance));
    }
}

#if defined(ARMCOMPUTENEON_ENABLED)
// ReLu fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseReLUIntoConvFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoDWConvFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<DepthwiseConvolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoFullyConnectedFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<FullyConnectedTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoBatchNormFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<BatchNormTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}

// BoundedReLu fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoConvFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoDWConvFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest < DepthwiseConvolution2dTest < DataType::Float32 > , DataType::Float32 >
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoFullyConnectedFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<FullyConnectedTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoBatchNormFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<BatchNormTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}

// ReLU fused into Receiver Layers QAsymmU8
BOOST_AUTO_TEST_CASE(FuseReLUIntoConvQAsymmU8CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::QAsymmU8>, DataType::QAsymmU8>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoDWConvQAsymmU8CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<DepthwiseConvolution2dTest<DataType::QAsymmU8>, DataType::QAsymmU8>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoFullyConnectedQAsymmU8CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<FullyConnectedTest<DataType::QAsymmU8>, DataType::QAsymmU8>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}

// HardSwish fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseHardSwishIntoConvFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::HardSwish;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}

// TanH fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseTanHIntoConvFloat32CpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::TanH;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::CpuAcc);
}
#endif

#if defined(ARMCOMPUTECL_ENABLED)
// ReLu fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseReLUIntoConvFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoDWConvFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<DepthwiseConvolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoFullyConnectedFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<FullyConnectedTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoBatchNormFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<BatchNormTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoMulFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<MultiplicationTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoAddFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<AdditionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoSubFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<SubtractionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUIntoDivFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<DivisionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}

// BoundedReLu fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoConvFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoDWConvFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<DepthwiseConvolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoFullyConnectedFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<FullyConnectedTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoBatchNormFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<BatchNormTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoMulFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<MultiplicationTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoAddFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<AdditionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoSubFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<SubtractionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseBoundedReLUIntoDivFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 1.0f;
    activationDescriptor.m_B = -1.0f;

    FuseActivationIntoPreviousLayerTest<DivisionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}

// ReLU fused into Receiver Layers QAsymmU8
BOOST_AUTO_TEST_CASE(FuseReLUQIntoConvAsymmU8GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::QAsymmU8>, DataType::QAsymmU8>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUQIntoDWConvAsymmU8GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<DepthwiseConvolution2dTest<DataType::QAsymmU8>, DataType::QAsymmU8>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseReLUQIntoFullyConnectedAsymmU8GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    FuseActivationIntoPreviousLayerTest<FullyConnectedTest<DataType::QAsymmU8>, DataType::QAsymmU8>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}

// HardSwish fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseHardSwishIntoConvFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::HardSwish;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseHardSwishIntoMulFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::HardSwish;

    FuseActivationIntoPreviousLayerTest<MultiplicationTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseHardSwishIntoAddFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::HardSwish;

    FuseActivationIntoPreviousLayerTest<AdditionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseHardSwishIntoSubFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::HardSwish;

    FuseActivationIntoPreviousLayerTest<SubtractionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseHardSwishIntoDivFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::HardSwish;

    FuseActivationIntoPreviousLayerTest<DivisionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}

// TanH fused into Receiver Layers Float32
BOOST_AUTO_TEST_CASE(FuseTanHIntoConvFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::TanH;

    FuseActivationIntoPreviousLayerTest<Convolution2dTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseTanHIntoMulFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::TanH;

    FuseActivationIntoPreviousLayerTest<MultiplicationTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseTanHIntoAddFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::TanH;

    FuseActivationIntoPreviousLayerTest<AdditionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseTanHIntoSubFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::TanH;

    FuseActivationIntoPreviousLayerTest<SubtractionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
BOOST_AUTO_TEST_CASE(FuseTanHIntoDivFloat32GpuAccTest)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::TanH;

    FuseActivationIntoPreviousLayerTest<DivisionTest<DataType::Float32>, DataType::Float32>
        (activationDescriptor, 0.0001f, armnn::Compute::GpuAcc);
}
#endif

BOOST_AUTO_TEST_SUITE_END()