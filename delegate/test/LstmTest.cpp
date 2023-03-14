//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LstmTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>
#include <doctest/doctest.h>

namespace armnnDelegate
{

void LstmTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 2;
    int32_t inputSize = 2;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    std::vector<int32_t> inputShape {batchSize , inputSize};
    std::vector<int32_t> cellStateInTensorInfo {batchSize , numUnits};
    std::vector<int32_t> outputStateInTensorInfo {batchSize , outputSize};

    std::vector<int32_t> scratchBufferTensorInfo {batchSize, numUnits * 4};
    std::vector<int32_t> cellStateOutTensorInfo {batchSize, numUnits};
    std::vector<int32_t> outputStateOutTensorInfo {batchSize, outputSize};
    std::vector<int32_t> outputTensorInfo {batchSize, outputSize};

    std::vector<int32_t> tensorInfo4 {numUnits};
    std::vector<int32_t> tensorInfo8 {numUnits, 2};
    std::vector<int32_t> tensorInfo16 {numUnits, 4};

    //tensorInfo8,
    bool hasInputToInputWeights = true;
    std::vector<float> inputToInputWeights {-0.45018822f, -0.02338299f, -0.0870589f,
                                            -0.34550029f, 0.04266912f, -0.15680569f,
                                            -0.34856534f, 0.43890524f};

    std::vector<float> inputToForgetWeights {0.09701663f, 0.20334584f, -0.50592935f,
                                             -0.31343272f, -0.40032279f, 0.44781327f,
                                             0.01387155f, -0.35593212f};

    std::vector<float> inputToCellWeights {-0.50013041f, 0.1370284f, 0.11810488f, 0.2013163f,
                                           -0.20583314f, 0.44344562f, 0.22077113f,
                                           -0.29909778f};

    std::vector<float> inputToOutputWeights {-0.25065863f, -0.28290087f, 0.04613829f,
                                             0.40525138f, 0.44272184f, 0.03897077f,
                                             -0.1556896f, 0.19487578f};

    //tensorInfo16,
    bool hasRecurrentToInputWeights = true;
    std::vector<float> recurrentToInputWeights {-0.0063535f, -0.2042388f, 0.31454784f,
                                                -0.35746509f, 0.28902304f, 0.08183324f,
                                                -0.16555229f, 0.02286911f, -0.13566875f,
                                                0.03034258f, 0.48091322f, -0.12528998f,
                                                0.24077177f, -0.51332325f, -0.33502164f,
                                                0.10629296f};

    std::vector<float> recurrentToForgetWeights {-0.48684245f, -0.06655136f, 0.42224967f,
                                                 0.2112639f, 0.27654213f, 0.20864892f,
                                                 -0.07646349f, 0.45877004f, 0.00141793f,
                                                 -0.14609534f, 0.36447752f, 0.09196436f,
                                                 0.28053468f, 0.01560611f, -0.20127171f,
                                                 -0.01140004f};

    std::vector<float> recurrentToCellWeights {-0.3407414f, 0.24443203f, -0.2078532f,
                                               0.26320225f, 0.05695659f, -0.00123841f,
                                               -0.4744786f, -0.35869038f, -0.06418842f,
                                               -0.13502428f, -0.501764f, 0.22830659f,
                                               -0.46367589f, 0.26016325f, -0.03894562f,
                                               -0.16368064f};

    std::vector<float> recurrentToOutputWeights {0.43385774f, -0.17194885f, 0.2718237f,
                                                 0.09215671f, 0.24107647f, -0.39835793f,
                                                 0.18212086f, 0.01301402f, 0.48572797f,
                                                 -0.50656658f, 0.20047462f, -0.20607421f,
                                                 -0.51818722f, -0.15390486f, 0.0468148f,
                                                 0.39922136f};
    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<float> cellToInputWeights {};
    bool hasCellToForgetWeights = false;
    std::vector<float> cellToForgetWeights {};
    bool hasCellToOutputWeights = false;
    std::vector<float> cellToOutputWeights {};

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias {0., 0., 0., 0.};
    std::vector<float> forgetGateBias {1., 1., 1., 1.};
    std::vector<float> cellBias {0., 0., 0., 0.};
    std::vector<float> outputGateBias {0., 0., 0., 0.};

    bool hasProjectionWeights = false;
    std::vector<float> projectionWeights;
    bool hasProjectionBias = false;
    std::vector<float> projectionBias;

    bool hasInputLayerNormWeights = false;
    std::vector<float> inputLayerNormWeights;
    bool hasForgetLayerNormWeights = false;
    std::vector<float> forgetLayerNormWeights;
    bool hasCellLayerNormWeights = false;
    std::vector<float> cellLayerNormWeights;
    bool hasOutputLayerNormWeights = false;
    std::vector<float> outputLayerNormWeights;

    std::vector<float> inputValues {2., 3., 3., 4.};
    std::vector<float> expectedOutputValues {-0.02973187f, 0.1229473f,   0.20885126f, -0.15358765f,
                                             -0.0185422f,   0.11281417f,  0.24466537f, -0.1826292f};

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 0.f;
    float clippingThresProj = 0.f;

    LstmTestImpl<float>(backends,
                        ::tflite::TensorType_FLOAT32,
                        batchSize,
                        inputSize,
                        outputSize,
                        numUnits,
                        hasInputToInputWeights,
                        inputToInputWeights,
                        inputToForgetWeights,
                        inputToCellWeights,
                        inputToOutputWeights,
                        hasRecurrentToInputWeights,
                        recurrentToInputWeights,
                        recurrentToForgetWeights,
                        recurrentToCellWeights,
                        recurrentToOutputWeights,
                        hasCellToInputWeights,
                        cellToInputWeights,
                        hasCellToForgetWeights,
                        cellToForgetWeights,
                        hasCellToOutputWeights,
                        cellToOutputWeights,
                        hasInputGateBias,
                        inputGateBias,
                        forgetGateBias,
                        cellBias,
                        outputGateBias,
                        hasProjectionWeights,
                        projectionWeights,
                        hasProjectionBias,
                        projectionBias,
                        hasInputLayerNormWeights,
                        inputLayerNormWeights,
                        hasForgetLayerNormWeights,
                        forgetLayerNormWeights,
                        hasCellLayerNormWeights,
                        cellLayerNormWeights,
                        hasOutputLayerNormWeights,
                        outputLayerNormWeights,
                        inputValues,
                        expectedOutputValues,
                        activationFunction,
                        clippingThresCell,
                        clippingThresProj);
}

TEST_SUITE("LstmTest_CpuRefTests")
{

TEST_CASE ("LstmTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    LstmTest(backends);
}

} //End of TEST_SUITE("Convolution2dTest_CpuRef")

TEST_SUITE("LstmTest_CpuAccTests")
{

TEST_CASE ("LstmTest_CpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    LstmTest(backends);
}

} //End of TEST_SUITE("Convolution2dTest_CpuAcc")

} // namespace armnnDelegate