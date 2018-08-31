//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/Workload.hpp>
#include <backends/RefWorkloads.hpp>
#include <backends/RefWorkloadFactory.hpp>

#include <armnn/Exceptions.hpp>

#include "WorkloadTestUtils.hpp"

using namespace armnn;

BOOST_AUTO_TEST_SUITE(WorkloadInfoValidation)



BOOST_AUTO_TEST_CASE(QueueDescriptor_Validate_WrongNumOfInputsOutputs)
{
    InputQueueDescriptor invalidData;
    WorkloadInfo invalidInfo;
    //Invalid argument exception is expected, because no inputs and no outputs were defined.
    BOOST_CHECK_THROW(RefWorkloadFactory().CreateInput(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(RefPooling2dFloat32Workload_Validate_WrongDimTensor)
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
    BOOST_CHECK_THROW(RefPooling2dFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(SoftmaxQueueDescriptor_Validate_WrongInputHeight)
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
    BOOST_CHECK_THROW(RefSoftmaxFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(FullyConnectedQueueDescriptor_Validate_RequiredDataMissing)
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

    ScopedCpuTensorHandle weightTensor(weightsDesc);
    ScopedCpuTensorHandle biasTensor(biasesDesc);

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);
    invalidData.m_Weight = &weightTensor;
    invalidData.m_Bias = &biasTensor;
    invalidData.m_Parameters.m_BiasEnabled = true;
    invalidData.m_Parameters.m_TransposeWeightMatrix = false;


    //Invalid argument exception is expected, because not all required fields have been provided.
    //In particular inputsData[0], outputsData[0] and weightsData can not be null.
    BOOST_CHECK_THROW(RefFullyConnectedFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}


BOOST_AUTO_TEST_CASE(NormalizationQueueDescriptor_Validate_WrongInputHeight)
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
    BOOST_CHECK_THROW(RefNormalizationFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(SplitterQueueDescriptor_Validate_WrongWindow)
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

    BOOST_TEST_INFO("Invalid argument exception is expected, because split window dimensionality does not "
        "match input.");
    BOOST_CHECK_THROW(RefSplitterFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);

    // Invalid, since window extends past the boundary of input tensor.
    std::vector<unsigned int> wOrigin3 = {0, 0, 15, 0};
    armnn::SplitterQueueDescriptor::ViewOrigin window3(wOrigin3);
    invalidData.m_ViewOrigins[0] = window3;
    BOOST_TEST_INFO("Invalid argument exception is expected (wOrigin3[2]+ outputHeight > inputHeight");
    BOOST_CHECK_THROW(RefSplitterFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);


    std::vector<unsigned int> wOrigin4 = {0, 0, 0, 0};
    armnn::SplitterQueueDescriptor::ViewOrigin window4(wOrigin4);
    invalidData.m_ViewOrigins[0] = window4;

    std::vector<unsigned int> wOrigin5 = {1, 16, 20, 2};
    armnn::SplitterQueueDescriptor::ViewOrigin window5(wOrigin4);
    invalidData.m_ViewOrigins.push_back(window5);

    BOOST_TEST_INFO("Invalid exception due to number of split windows not matching number of outputs.");
    BOOST_CHECK_THROW(RefSplitterFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}


BOOST_AUTO_TEST_CASE(MergerQueueDescriptor_Validate_WrongWindow)
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

    MergerQueueDescriptor invalidData;
    WorkloadInfo          invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    // Invalid, since it has only 3 dimensions while the input tensor is 4d.
    std::vector<unsigned int> wOrigin = {0, 0, 0};
    armnn::MergerQueueDescriptor::ViewOrigin window(wOrigin);
    invalidData.m_ViewOrigins.push_back(window);

    BOOST_TEST_INFO("Invalid argument exception is expected, because merge window dimensionality does not "
        "match input.");
    BOOST_CHECK_THROW(RefMergerFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);

    // Invalid, since window extends past the boundary of output tensor.
    std::vector<unsigned int> wOrigin3 = {0, 0, 15, 0};
    armnn::MergerQueueDescriptor::ViewOrigin window3(wOrigin3);
    invalidData.m_ViewOrigins[0] = window3;
    BOOST_TEST_INFO("Invalid argument exception is expected (wOrigin3[2]+ inputHeight > outputHeight");
    BOOST_CHECK_THROW(RefMergerFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);


    std::vector<unsigned int> wOrigin4 = {0, 0, 0, 0};
    armnn::MergerQueueDescriptor::ViewOrigin window4(wOrigin4);
    invalidData.m_ViewOrigins[0] = window4;

    std::vector<unsigned int> wOrigin5 = {1, 16, 20, 2};
    armnn::MergerQueueDescriptor::ViewOrigin window5(wOrigin4);
    invalidData.m_ViewOrigins.push_back(window5);

    BOOST_TEST_INFO("Invalid exception due to number of merge windows not matching number of inputs.");
    BOOST_CHECK_THROW(RefMergerFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(AdditionQueueDescriptor_Validate_InputNumbers)
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
    BOOST_CHECK_THROW(RefAdditionFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);

    AddInputToWorkload(invalidData, invalidInfo, input2TensorInfo, nullptr);

    // Correct.
    BOOST_CHECK_NO_THROW(RefAdditionFloat32Workload(invalidData, invalidInfo));

    AddInputToWorkload(invalidData, invalidInfo, input3TensorInfo, nullptr);

    // Too many inputs.
    BOOST_CHECK_THROW(RefAdditionFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(AdditionQueueDescriptor_Validate_InputShapes)
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

        BOOST_CHECK_THROW(RefAdditionFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
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
        BOOST_CHECK_THROW(RefAdditionFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
    }
}

BOOST_AUTO_TEST_CASE(MultiplicationQueueDescriptor_Validate_InputTensorDimensionMismatch)
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

        BOOST_CHECK_THROW(RefMultiplicationFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
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

        BOOST_CHECK_THROW(RefMultiplicationFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
    }
}

BOOST_AUTO_TEST_CASE(ReshapeQueueDescriptor_Validate_MismatchingNumElements)
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
    BOOST_CHECK_THROW(RefReshapeFloat32Workload(invalidData, invalidInfo), armnn::InvalidArgumentException);
}


BOOST_AUTO_TEST_CASE(LstmQueueDescriptor_Validate)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 1, 2 };
    unsigned int outputShape[] = { 1 };

    inputTensorInfo = armnn::TensorInfo(2, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(1, outputShape, armnn::DataType::Float32);

    LstmQueueDescriptor invalidData;
    WorkloadInfo        invalidInfo;

    AddInputToWorkload(invalidData, invalidInfo, inputTensorInfo, nullptr);
    AddOutputToWorkload(invalidData, invalidInfo, outputTensorInfo, nullptr);

    BOOST_CHECK_THROW(invalidData.Validate(invalidInfo), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_SUITE_END()
