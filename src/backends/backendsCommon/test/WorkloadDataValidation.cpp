//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnn/Exceptions.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/Workload.hpp>

#include <reference/workloads/RefWorkloads.hpp>
#include <reference/RefWorkloadFactory.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("WorkloadInfoValidation")
{
TEST_CASE("BatchNormalizationQueueDescriptor_Validate_DifferentQuantizationData")
{
    TensorShape inputShape { 1, 3, 2, 2 };
    TensorShape outputShape { 1, 3, 2, 2 };

    TensorInfo inputTensorInfo(inputShape, armnn::DataType::QAsymmU8, .1f, 125);
    TensorInfo outputTensorInfo(outputShape, armnn::DataType::QAsymmU8, .2f, 120);

    BatchNormalizationQueueDescriptor invalidData;
    WorkloadInfo                      invalidInfo;

    unsigned int sameShape[] = { 10 };
    TensorInfo sameInfo = armnn::TensorInfo(1, sameShape, armnn::DataType::QAsymmU8);
    ScopedTensorHandle sameTensor(sameInfo);

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    invalidData.m_Mean = &sameTensor;
    invalidData.m_Variance = &sameTensor;
    invalidData.m_Beta= &sameTensor;
    invalidData.m_Gamma = &sameTensor;

    CHECK_NOTHROW(RefBatchNormalizationWorkload(invalidData, invalidInfo));
}

TEST_CASE("QueueDescriptor_Validate_WrongNumOfInputsOutputs")
{
    InputQueueDescriptor invalidData;
    WorkloadInfo invalidInfo;
    //Invalid argument exception is expected, because no inputs and no outputs were defined.
    CHECK_THROWS_AS(RefWorkloadFactory().CreateWorkload(LayerType::Input, invalidData, invalidInfo),
                    armnn::InvalidArgumentException);
}

TEST_CASE("RefPooling2dFloat32Workload_Validate_WrongDimTensor")
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {2, 3, 4}; // <- Invalid - input tensor has to be 4D.
    unsigned int outputShape[] = {2, 3, 4, 5};

    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);
    inputTensorInfo  = armnn::TensorInfo(3, inputShape, armnn::DataType::Float32);

    Pooling2dQueueDescriptor invalidData;
    WorkloadInfo           invalidInfo;

    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);

    // Invalid argument exception is expected, input tensor has to be 4D.
    CHECK_THROWS_AS(RefPooling2dWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

TEST_CASE("RefPooling3dFloat32Workload_Validate_WrongDimTensor")
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {2, 3, 4, 5}; // <- Invalid - input tensor has to be 5D.
    unsigned int outputShape[] = {2, 3, 4, 5, 6};

    outputTensorInfo = armnn::TensorInfo(5, outputShape, armnn::DataType::Float32);
    inputTensorInfo  = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);

    Pooling3dQueueDescriptor invalidData;
    WorkloadInfo           invalidInfo;

    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);

    // Invalid argument exception is expected, input tensor has to be 5D.
    CHECK_THROWS_AS(RefPooling3dWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

TEST_CASE("SoftmaxQueueDescriptor_Validate_WrongInputHeight")
{
    unsigned int inputHeight = 1;
    unsigned int inputWidth = 1;
    unsigned int inputChannels = 4;
    unsigned int inputNum = 2;

    unsigned int outputChannels = inputChannels;
    unsigned int outputHeight = inputHeight + 1;    //Makes data invalid - Softmax expects height and width to be 1.
    unsigned int outputWidth = inputWidth;
    unsigned int outputNum = inputNum;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[] = { outputNum, outputChannels, outputHeight, outputWidth };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    SoftmaxQueueDescriptor invalidData;
    WorkloadInfo           invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    //Invalid argument exception is expected, because height != 1.
    CHECK_THROWS_AS(RefSoftmaxWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

TEST_CASE("FullyConnectedQueueDescriptor_Validate_RequiredDataMissing")
{
    unsigned int inputWidth = 1;
    unsigned int inputHeight = 1;
    unsigned int inputChannels = 5;
    unsigned int inputNum = 2;

    unsigned int outputWidth = 1;
    unsigned int outputHeight = 1;
    unsigned int outputChannels = 3;
    unsigned int outputNum = 2;

    // Define the tensor descriptors.
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;
    armnn::TensorInfo weightsDesc;
    armnn::TensorInfo biasesDesc;

    unsigned int inputShape[] = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[] = { outputNum, outputChannels, outputHeight, outputWidth };
    unsigned int weightsShape[] = { 1, 1, inputChannels, outputChannels };
    unsigned int biasShape[] = { 1, outputChannels, outputHeight, outputWidth };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);
    weightsDesc = armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32);
    biasesDesc = armnn::TensorInfo(4, biasShape, armnn::DataType::Float32);

    FullyConnectedQueueDescriptor invalidData;
    WorkloadInfo                  invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
    invalidData.m_Parameters.m_BiasEnabled = true;
    invalidData.m_Parameters.m_TransposeWeightMatrix = false;


    //Invalid argument exception is expected, because not all required fields have been provided.
    //In particular inputsData[0], outputsData[0] and weightsData can not be null.
    CHECK_THROWS_AS(RefFullyConnectedWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}


TEST_CASE("NormalizationQueueDescriptor_Validate_WrongInputHeight")
{
    constexpr unsigned int inputNum = 5;
    constexpr unsigned int inputHeight   = 32;
    constexpr unsigned int inputWidth    = 24;
    constexpr unsigned int inputChannels = 3;

    constexpr unsigned int outputNum = inputNum;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputHeight = inputHeight + 1; //Makes data invalid - normalization requires.
                                                           //Input and output to have the same dimensions.
    constexpr unsigned int outputWidth  = inputWidth;


    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {inputNum, inputChannels, inputHeight, inputWidth};
    unsigned int outputShape[] = {outputNum, outputChannels, outputHeight, outputWidth};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);


    armnn::NormalizationAlgorithmMethod normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    armnn::NormalizationAlgorithmChannel normChannel = armnn::NormalizationAlgorithmChannel::Across;
    float alpha = 1.f;
    float beta = 1.f;
    float kappa = 1.f;
    uint32_t normSize = 5;

    NormalizationQueueDescriptor invalidData;
    WorkloadInfo                 invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
    invalidData.m_Parameters.m_NormChannelType = normChannel;
    invalidData.m_Parameters.m_NormMethodType  = normMethod;
    invalidData.m_Parameters.m_NormSize        = normSize;
    invalidData.m_Parameters.m_Alpha           = alpha;
    invalidData.m_Parameters.m_Beta            = beta;
    invalidData.m_Parameters.m_K               = kappa;

    //Invalid argument exception is expected, because input height != output height.
    CHECK_THROWS_AS(RefNormalizationWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

TEST_CASE("SplitterQueueDescriptor_Validate_WrongWindow")
{
    constexpr unsigned int inputNum = 1;
    constexpr unsigned int inputHeight   = 32;
    constexpr unsigned int inputWidth    = 24;
    constexpr unsigned int inputChannels = 3;

    constexpr unsigned int outputNum = inputNum;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputHeight = 18;
    constexpr unsigned int outputWidth  = inputWidth;


    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {inputNum, inputChannels, inputHeight, inputWidth};
    unsigned int outputShape[] = {outputNum, outputChannels, outputHeight, outputWidth};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    SplitterQueueDescriptor invalidData;
    WorkloadInfo            invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    // Invalid, since it has only 3 dimensions while the input tensor is 4d.
    std::vector<unsigned int> wOrigin = {0, 0, 0};
    armnn::SplitterQueueDescriptor::ViewOrigin window(wOrigin);
    invalidData.m_ViewOrigins.push_back(window);

    INFO("Invalid argument exception is expected, because split window dimensionality does not match input.");
    CHECK_THROWS_AS(RefSplitterWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);

    // Invalid, since window extends past the boundary of input tensor.
    std::vector<unsigned int> wOrigin3 = {0, 0, 15, 0};
    armnn::SplitterQueueDescriptor::ViewOrigin window3(wOrigin3);
    invalidData.m_ViewOrigins[0] = window3;
    INFO("Invalid argument exception is expected (wOrigin3[2]+ outputHeight > inputHeight");
    CHECK_THROWS_AS(RefSplitterWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);


    std::vector<unsigned int> wOrigin4 = {0, 0, 0, 0};
    armnn::SplitterQueueDescriptor::ViewOrigin window4(wOrigin4);
    invalidData.m_ViewOrigins[0] = window4;

    std::vector<unsigned int> wOrigin5 = {1, 16, 20, 2};
    armnn::SplitterQueueDescriptor::ViewOrigin window5(wOrigin4);
    invalidData.m_ViewOrigins.push_back(window5);

    INFO("Invalid exception due to number of split windows not matching number of outputs.");
    CHECK_THROWS_AS(RefSplitterWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}


TEST_CASE("ConcatQueueDescriptor_Validate_WrongWindow")
{
    constexpr unsigned int inputNum = 1;
    constexpr unsigned int inputChannels = 3;
    constexpr unsigned int inputHeight   = 32;
    constexpr unsigned int inputWidth    = 24;

    constexpr unsigned int outputNum = 1;
    constexpr unsigned int outputChannels = 3;
    constexpr unsigned int outputHeight = 32;
    constexpr unsigned int outputWidth  = 24;


    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {inputNum, inputChannels, inputHeight, inputWidth};
    unsigned int outputShape[] = {outputNum, outputChannels, outputHeight, outputWidth};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    ConcatQueueDescriptor invalidData;
    WorkloadInfo          invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    // Invalid, since it has only 3 dimensions while the input tensor is 4d.
    std::vector<unsigned int> wOrigin = {0, 0, 0};
    armnn::ConcatQueueDescriptor::ViewOrigin window(wOrigin);
    invalidData.m_ViewOrigins.push_back(window);

    INFO("Invalid argument exception is expected, because merge window dimensionality does not match input.");
    CHECK_THROWS_AS(RefConcatWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);

    // Invalid, since window extends past the boundary of output tensor.
    std::vector<unsigned int> wOrigin3 = {0, 0, 15, 0};
    armnn::ConcatQueueDescriptor::ViewOrigin window3(wOrigin3);
    invalidData.m_ViewOrigins[0] = window3;
    INFO("Invalid argument exception is expected (wOrigin3[2]+ inputHeight > outputHeight");
    CHECK_THROWS_AS(RefConcatWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);


    std::vector<unsigned int> wOrigin4 = {0, 0, 0, 0};
    armnn::ConcatQueueDescriptor::ViewOrigin window4(wOrigin4);
    invalidData.m_ViewOrigins[0] = window4;

    std::vector<unsigned int> wOrigin5 = {1, 16, 20, 2};
    armnn::ConcatQueueDescriptor::ViewOrigin window5(wOrigin4);
    invalidData.m_ViewOrigins.push_back(window5);

    INFO("Invalid exception due to number of merge windows not matching number of inputs.");
    CHECK_THROWS_AS(RefConcatWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

TEST_CASE("AdditionQueueDescriptor_Validate_InputNumbers")
{
    armnn::TensorInfo input1TensorInfo;
    armnn::TensorInfo input2TensorInfo;
    armnn::TensorInfo input3TensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape[]  = {1, 1, 1, 1};

    input1TensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    input2TensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    input3TensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);

    AdditionQueueDescriptor invalidData;
    WorkloadInfo            invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, input1TensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    // Too few inputs.
    CHECK_THROWS_AS(RefAdditionWorkload<>(invalidData, invalidInfo), armnn::InvalidArgumentException);

    AddInputToWorkload(invalidData, invalidInfo, input2TensorInfo, nullptr);

    // Correct.
    CHECK_NOTHROW(RefAdditionWorkload<>(invalidData, invalidInfo));

    AddInputToWorkload(invalidData, invalidInfo, input3TensorInfo, nullptr);

    // Too many inputs.
    CHECK_THROWS_AS(RefAdditionWorkload<>(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

TEST_CASE("AdditionQueueDescriptor_Validate_InputShapes")
{
    armnn::TensorInfo input1TensorInfo;
    armnn::TensorInfo input2TensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape1[] = {1, 1, 2, 1};
    unsigned int shape2[] = {1, 1, 3, 2};

    // Incompatible shapes even with broadcasting.
    {
        input1TensorInfo = armnn::TensorInfo(4, shape1, armnn::DataType::Float32);
        input2TensorInfo = armnn::TensorInfo(4, shape2, armnn::DataType::Float32);
        outputTensorInfo = armnn::TensorInfo(4, shape1, armnn::DataType::Float32);

        AdditionQueueDescriptor invalidData;
        WorkloadInfo            invalidInfo;

        AddInputToWorkload(invalidData, invalidInfo, input1TensorInfo, nullptr);
        AddInputToWorkload(invalidData, invalidInfo, input2TensorInfo, nullptr);
        AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

        CHECK_THROWS_AS(RefAdditionWorkload<>(invalidData, invalidInfo), armnn::InvalidArgumentException);
    }

    // Output size not compatible with input sizes.
    {
        input1TensorInfo = armnn::TensorInfo(4, shape1, armnn::DataType::Float32);
        input2TensorInfo = armnn::TensorInfo(4, shape1, armnn::DataType::Float32);
        outputTensorInfo = armnn::TensorInfo(4, shape2, armnn::DataType::Float32);

        AdditionQueueDescriptor invalidData;
        WorkloadInfo            invalidInfo;

        AddInputToWorkload(invalidData, invalidInfo, input1TensorInfo, nullptr);
        AddInputToWorkload(invalidData, invalidInfo, input2TensorInfo, nullptr);
        AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

        // Output differs.
        CHECK_THROWS_AS(RefAdditionWorkload<>(invalidData, invalidInfo), armnn::InvalidArgumentException);
    }
}

TEST_CASE("MultiplicationQueueDescriptor_Validate_InputTensorDimensionMismatch")
{
    armnn::TensorInfo input0TensorInfo;
    armnn::TensorInfo input1TensorInfo;
    armnn::TensorInfo outputTensorInfo;

    constexpr unsigned int input0Shape[] = { 2, 2, 4, 4 };
    constexpr std::size_t dimensionCount = std::extent<decltype(input0Shape)>::value;

    // Checks dimension consistency for input tensors.
    for (unsigned int dimIndex = 0; dimIndex < dimensionCount; ++dimIndex)
    {
        unsigned int input1Shape[dimensionCount];
        for (unsigned int i = 0; i < dimensionCount; ++i)
        {
            input1Shape[i] = input0Shape[i];
        }

        ++input1Shape[dimIndex];

        input0TensorInfo = armnn::TensorInfo(dimensionCount, input0Shape, armnn::DataType::Float32);
        input1TensorInfo = armnn::TensorInfo(dimensionCount, input1Shape, armnn::DataType::Float32);
        outputTensorInfo = armnn::TensorInfo(dimensionCount, input0Shape, armnn::DataType::Float32);

        MultiplicationQueueDescriptor invalidData;
        WorkloadInfo                  invalidInfo;

        AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
        AddInputToWorkload(invalidData, invalidInfo, input0TensorInfo, nullptr);
        AddInputToWorkload(invalidData, invalidInfo, input1TensorInfo, nullptr);

        CHECK_THROWS_AS(RefMultiplicationWorkload<>(invalidData, invalidInfo), armnn::InvalidArgumentException);
    }

    // Checks dimension consistency for input and output tensors.
    for (unsigned int dimIndex = 0; dimIndex < dimensionCount; ++dimIndex)
    {
        unsigned int outputShape[dimensionCount];
        for (unsigned int i = 0; i < dimensionCount; ++i)
        {
            outputShape[i] = input0Shape[i];
        }

        ++outputShape[dimIndex];

        input0TensorInfo = armnn::TensorInfo(dimensionCount, input0Shape, armnn::DataType::Float32);
        input1TensorInfo = armnn::TensorInfo(dimensionCount, input0Shape, armnn::DataType::Float32);
        outputTensorInfo = armnn::TensorInfo(dimensionCount, outputShape, armnn::DataType::Float32);

        MultiplicationQueueDescriptor invalidData;
        WorkloadInfo                  invalidInfo;

        AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
        AddInputToWorkload(invalidData, invalidInfo, input0TensorInfo, nullptr);
        AddInputToWorkload(invalidData, invalidInfo, input1TensorInfo, nullptr);

        CHECK_THROWS_AS(RefMultiplicationWorkload<>(invalidData, invalidInfo), armnn::InvalidArgumentException);
    }
}

TEST_CASE("ReshapeQueueDescriptor_Validate_MismatchingNumElements")
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    // The input and output shapes should have the same number of elements, but these don't.
    unsigned int inputShape[] = { 1, 1, 2, 3 };
    unsigned int outputShape[] = { 1, 1, 1, 2 };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    ReshapeQueueDescriptor invalidData;
    WorkloadInfo           invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    // InvalidArgumentException is expected, because the number of elements don't match.
    CHECK_THROWS_AS(RefReshapeWorkload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}


TEST_CASE("LstmQueueDescriptor_Validate")
{
    armnn::DataType dataType = armnn::DataType::Float32;

    float qScale = 1.0f;
    int32_t qOffset = 0;

    unsigned int batchSize = 2;
    unsigned int outputSize = 3;
    unsigned int inputSize = 5;
    unsigned numUnits = 4;

    armnn::TensorInfo inputTensorInfo({batchSize , inputSize}, dataType,  qScale, qOffset );
    armnn::TensorInfo outputStateInTensorInfo({batchSize , outputSize}, dataType, qScale, qOffset);
    armnn::TensorInfo cellStateInTensorInfo({batchSize , numUnits}, dataType, qScale, qOffset);

    // Scratch buffer size with CIFG [batchSize, numUnits * 4]
    armnn::TensorInfo scratchBufferTensorInfo({batchSize, numUnits * 4}, dataType, qScale, qOffset);
    armnn::TensorInfo cellStateOutTensorInfo({batchSize, numUnits}, dataType, qScale, qOffset);
    armnn::TensorInfo outputStateOutTensorInfo({batchSize, outputSize}, dataType, qScale, qOffset);
    armnn::TensorInfo outputTensorInfo({batchSize, outputSize}, dataType, qScale, qOffset);

    armnn::TensorInfo tensorInfo3({outputSize}, dataType, qScale, qOffset);
    armnn::TensorInfo tensorInfo4({numUnits}, dataType, qScale, qOffset);
    armnn::TensorInfo tensorInfo4x5({numUnits, inputSize}, dataType, qScale, qOffset);
    armnn::TensorInfo tensorInfo4x3({numUnits, outputSize}, dataType, qScale, qOffset);
    armnn::TensorInfo tensorInfo3x4({outputSize, numUnits}, dataType, qScale, qOffset);

    LstmQueueDescriptor data;
    WorkloadInfo        info;

    AddInputToWorkload(data, info, inputTensorInfo, nullptr);
    AddInputToWorkload(data, info, outputStateInTensorInfo, nullptr);
    AddInputToWorkload(data, info, cellStateInTensorInfo, nullptr);

    AddOutputToWorkload(data, info, scratchBufferTensorInfo, nullptr);
    AddOutputToWorkload(data, info, outputStateOutTensorInfo, nullptr);
    AddOutputToWorkload(data, info, cellStateOutTensorInfo, nullptr);
    // AddOutputToWorkload(data, info, outputTensorInfo, nullptr); is left out

    armnn::ScopedTensorHandle inputToInputWeightsTensor(tensorInfo4x5);
    armnn::ScopedTensorHandle inputToForgetWeightsTensor(tensorInfo4x5);
    armnn::ScopedTensorHandle inputToCellWeightsTensor(tensorInfo4x5);
    armnn::ScopedTensorHandle inputToOutputWeightsTensor(tensorInfo4x5);
    armnn::ScopedTensorHandle recurrentToForgetWeightsTensor(tensorInfo4x3);
    armnn::ScopedTensorHandle recurrentToInputWeightsTensor(tensorInfo4x3);
    armnn::ScopedTensorHandle recurrentToCellWeightsTensor(tensorInfo4x3);
    armnn::ScopedTensorHandle recurrentToOutputWeightsTensor(tensorInfo4x3);
    armnn::ScopedTensorHandle cellToInputWeightsTensor(tensorInfo4);
    armnn::ScopedTensorHandle inputGateBiasTensor(tensorInfo4);
    armnn::ScopedTensorHandle forgetGateBiasTensor(tensorInfo4);
    armnn::ScopedTensorHandle cellBiasTensor(tensorInfo4);
    armnn::ScopedTensorHandle outputGateBiasTensor(tensorInfo4);
    armnn::ScopedTensorHandle cellToForgetWeightsTensor(tensorInfo4);
    armnn::ScopedTensorHandle cellToOutputWeightsTensor(tensorInfo4);
    armnn::ScopedTensorHandle projectionWeightsTensor(tensorInfo3x4);
    armnn::ScopedTensorHandle projectionBiasTensor(tensorInfo3);
    armnn::ScopedTensorHandle inputLayerNormWeightsTensor(tensorInfo4);
    armnn::ScopedTensorHandle forgetLayerNormWeightsTensor(tensorInfo4);
    armnn::ScopedTensorHandle cellLayerNormWeightsTensor(tensorInfo4);
    armnn::ScopedTensorHandle outputLayerNormWeightsTensor(tensorInfo4);

    data.m_InputToInputWeights = &inputToInputWeightsTensor;
    data.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    data.m_InputToCellWeights = &inputToCellWeightsTensor;
    data.m_InputToOutputWeights = &inputToOutputWeightsTensor;
    data.m_RecurrentToInputWeights = &recurrentToInputWeightsTensor;
    data.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    data.m_RecurrentToCellWeights = &recurrentToCellWeightsTensor;
    data.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;
    data.m_CellToInputWeights = &cellToInputWeightsTensor;
    data.m_InputGateBias = &inputGateBiasTensor;
    data.m_ForgetGateBias = &forgetGateBiasTensor;
    data.m_CellBias = &cellBiasTensor;
    data.m_OutputGateBias = &outputGateBiasTensor;
    data.m_CellToForgetWeights = &cellToForgetWeightsTensor;
    data.m_CellToOutputWeights = &cellToOutputWeightsTensor;
    data.m_ProjectionWeights = &projectionWeightsTensor;
    data.m_ProjectionBias = &projectionBiasTensor;

    data.m_InputLayerNormWeights = &inputLayerNormWeightsTensor;
    data.m_ForgetLayerNormWeights = &forgetLayerNormWeightsTensor;
    data.m_CellLayerNormWeights = &cellLayerNormWeightsTensor;
    data.m_OutputLayerNormWeights = &outputLayerNormWeightsTensor;

    // Flags to set test configuration
    data.m_Parameters.m_ActivationFunc = 4;
    data.m_Parameters.m_CifgEnabled = false;
    data.m_Parameters.m_PeepholeEnabled = true;
    data.m_Parameters.m_ProjectionEnabled = true;
    data.m_Parameters.m_LayerNormEnabled = true;

    // check wrong number of outputs
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    AddOutputToWorkload(data, info, outputTensorInfo, nullptr);

    // check wrong cifg parameter configuration
    data.m_Parameters.m_CifgEnabled = true;
    armnn::TensorInfo scratchBufferTensorInfo2({batchSize, numUnits * 3}, dataType, qScale, qOffset);
    SetWorkloadOutput(data, info, 0, scratchBufferTensorInfo2, nullptr);
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_Parameters.m_CifgEnabled = false;
    SetWorkloadOutput(data, info, 0, scratchBufferTensorInfo, nullptr);

    // check wrong inputGateBias configuration
    data.m_InputGateBias = nullptr;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_InputGateBias = &inputGateBiasTensor;

    // check inconsistant projection parameters
    data.m_Parameters.m_ProjectionEnabled = false;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_Parameters.m_ProjectionEnabled = true;
    data.m_ProjectionWeights = nullptr;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_ProjectionWeights = &projectionWeightsTensor;

    // check missing input layer normalisation weights
    data.m_InputLayerNormWeights = nullptr;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_InputLayerNormWeights = &inputLayerNormWeightsTensor;

    // layer norm disabled but normalisation weights are present
    data.m_Parameters.m_LayerNormEnabled = false;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_Parameters.m_LayerNormEnabled = true;

    // check invalid outputTensor shape
    armnn::TensorInfo incorrectOutputTensorInfo({batchSize, outputSize + 1}, dataType, qScale, qOffset);
    SetWorkloadOutput(data, info, 3, incorrectOutputTensorInfo, nullptr);
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    SetWorkloadOutput(data, info, 3, outputTensorInfo, nullptr);

    // check invalid cell clipping parameters
    data.m_Parameters.m_ClippingThresCell = -1.0f;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_Parameters.m_ClippingThresCell = 0.0f;

    // check invalid projection clipping parameters
    data.m_Parameters.m_ClippingThresProj = -1.0f;
    CHECK_THROWS_AS(data.Validate(info), armnn::InvalidArgumentException);
    data.m_Parameters.m_ClippingThresProj = 0.0f;

    // check correct configuration
    CHECK_NOTHROW(data.Validate(info));
}

TEST_CASE("BiasPerAxisQuantization_ValidateCorrectValues")
{
    constexpr unsigned int nInput  = 1u;
    constexpr unsigned int cInput  = 3u;
    constexpr unsigned int hInput  = 3u;
    constexpr unsigned int wInput  = 3u;

    constexpr unsigned int nOutput = nInput;
    constexpr unsigned int cOutput = cInput;
    constexpr unsigned int hOutput = 1u;
    constexpr unsigned int wOutput = 1u;

    const TensorShape inputShape { nInput,  cInput,  hInput,  wInput  };
    const TensorShape outputShape{ nOutput, cOutput, hOutput, wOutput };
    const TensorShape weightShape{ cOutput, cInput,  hInput,  wInput  };
    const TensorShape biasShape  { cOutput                            };

    constexpr DataType inputType  = DataType::QAsymmU8;
    constexpr DataType weightType = DataType::QSymmS8;
    constexpr DataType biasType   = DataType::Signed32;

    constexpr float perTensorScale = 1.5f;
    const TensorInfo inputInfo (inputShape,  inputType, perTensorScale);
    const TensorInfo outputInfo(outputShape, inputType, perTensorScale);

    const std::vector<float> weightPerAxisScales = { 2.50f, 3.50f };
    const TensorInfo weightInfo(weightShape, weightType, weightPerAxisScales, 0);

    Convolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters.m_BiasEnabled = true;

    WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, nullptr);
    AddInputToWorkload(queueDescriptor, workloadInfo, weightInfo, nullptr);
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, nullptr);

    // Test 1: correct per-axis quantization values
    const std::vector<float> biasPerAxisScales1  = { 3.75f, 5.25f };
    const TensorInfo biasInfo1(biasShape, biasType, biasPerAxisScales1, 0);

    AddInputToWorkload(queueDescriptor, workloadInfo, biasInfo1, nullptr);

    CHECK_NOTHROW(queueDescriptor.Validate(workloadInfo));
}

TEST_CASE("BiasPerAxisQuantization_ValidateIncorrectValues")
{
    constexpr unsigned int nInput  = 1u;
    constexpr unsigned int cInput  = 3u;
    constexpr unsigned int hInput  = 3u;
    constexpr unsigned int wInput  = 3u;

    constexpr unsigned int nOutput = nInput;
    constexpr unsigned int cOutput = cInput;
    constexpr unsigned int hOutput = 1u;
    constexpr unsigned int wOutput = 1u;

    const TensorShape inputShape { nInput,  cInput,  hInput,  wInput  };
    const TensorShape outputShape{ nOutput, cOutput, hOutput, wOutput };
    const TensorShape weightShape{ cOutput, cInput,  hInput,  wInput  };
    const TensorShape biasShape  { cOutput                            };

    constexpr DataType inputType  = DataType::QAsymmU8;
    constexpr DataType weightType = DataType::QSymmS8;
    constexpr DataType biasType   = DataType::Signed32;

    constexpr float perTensorScale = 1.5f;
    const TensorInfo inputInfo (inputShape,  inputType, perTensorScale);
    const TensorInfo outputInfo(outputShape, inputType, perTensorScale);

    const std::vector<float> weightPerAxisScales = { 2.50f, 3.50f };
    const TensorInfo weightInfo(weightShape, weightType, weightPerAxisScales, 0);

    Convolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters.m_BiasEnabled = true;

    WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, nullptr);
    AddInputToWorkload(queueDescriptor, workloadInfo, weightInfo, nullptr);
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, nullptr);

    // Test 2: wrong per-axis quantization values
    const std::vector<float> biasPerAxisScales2 = { 4.00f, 5.00f };
    const TensorInfo biasInfo2(biasShape, biasType, biasPerAxisScales2, 0);

    AddInputToWorkload(queueDescriptor, workloadInfo, biasInfo2, nullptr);

    CHECK_NOTHROW(queueDescriptor.Validate(workloadInfo));

}

TEST_CASE("BiasPerAxisQuantization_ValidateInvalidArgumentException")
{
    constexpr unsigned int nInput  = 1u;
    constexpr unsigned int cInput  = 3u;
    constexpr unsigned int hInput  = 3u;
    constexpr unsigned int wInput  = 3u;

    constexpr unsigned int nOutput = nInput;
    constexpr unsigned int cOutput = cInput;
    constexpr unsigned int hOutput = 1u;
    constexpr unsigned int wOutput = 1u;

    const TensorShape inputShape { nInput,  cInput,  hInput,  wInput  };
    const TensorShape outputShape{ nOutput, cOutput, hOutput, wOutput };
    const TensorShape weightShape{ cOutput, cInput,  hInput,  wInput  };
    const TensorShape biasShape  { cOutput                            };

    constexpr DataType inputType  = DataType::QAsymmU8;
    constexpr DataType weightType = DataType::QSymmS8;
    constexpr DataType biasType   = DataType::Signed32;

    constexpr float perTensorScale = 1.5f;
    const TensorInfo inputInfo (inputShape,  inputType, perTensorScale);
    const TensorInfo outputInfo(outputShape, inputType, perTensorScale);

    const std::vector<float> weightPerAxisScales = { 2.50f, 3.50f };
    const TensorInfo weightInfo(weightShape, weightType, weightPerAxisScales, 0);

    Convolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters.m_BiasEnabled = true;

    WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, nullptr);
    AddInputToWorkload(queueDescriptor, workloadInfo, weightInfo, nullptr);
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, nullptr);

    // Test 3: mismatched number of quantization scales
    const std::vector<float> biasPerAxisScales3 = { 3.75f, 5.25f, 5.25f };
    const TensorInfo biasInfo3(biasShape, biasType, biasPerAxisScales3, 0);

    AddInputToWorkload(queueDescriptor, workloadInfo, biasInfo3, nullptr);

    CHECK_THROWS_AS(queueDescriptor.Validate(workloadInfo), InvalidArgumentException);
}


}
