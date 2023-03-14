//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "UnidirectionalSequenceLstmTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>
#include <doctest/doctest.h>

namespace armnnDelegate
{

void UnidirectionalSequenceLstmTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    //tensorInfo12,
    bool hasInputToInputWeights = true;
    std::vector<float> inputToInputWeights = { -0.49536117f, -0.0556083915f, -0.102400711f,
                                               -0.117484632f, 0.3298470976f, -0.1179017122f,
                                               0.214305695f, 0.42135173085f, 0.003878414626f,
                                               -0.348303917f, -0.1881275477f, 0.0343011027f };

    std::vector<float> inputToForgetWeights = { 0.2415594226f, 0.15400093799f, 0.4566498398f,
                                                -0.3810434485f, 0.268383264f, -0.009807467424f,
                                                -0.3522925403f, -0.24275735512f, -0.28344226125f,
                                                0.13512269116f, -0.4932442977f, -0.10039821991f };

    std::vector<float> inputToCellWeights = { -0.2504855627f, 0.184490025045f, -0.2480507493f,
                                              0.386399507f, -0.259465157985f, -0.16545993089f,
                                              -0.4230232555f, 0.341664791103f, -0.18127849691f,
                                              -0.2277662414f, -0.55275535589f, 0.34184026718f };

    std::vector<float> inputToOutputWeights = { 0.2303854227f, 0.5218806862f, -0.4865379333f,
                                                0.53969591851f, 0.23393625035f, -0.27140527306f,
                                                0.50009280443f, 0.07511717046f, 0.3998299249f,
                                                -0.51717478049f, 0.1889653282f, -0.367323637f };

    //tensorInfo16,
    bool hasRecurrentToInputWeights = true;
    std::vector<float> recurrentToInputWeights = { -0.128009796112f, 0.1995525098f, -0.07745539397f, 0.1558421701f,
                                                   -0.265254765766f, -0.38837709614f, -0.05636804124f, 0.4259087456f,
                                                   0.17628988623f, 0.3877420127f, 0.53300309181f, -0.0959980934f,
                                                   0.00302857416f, 0.3266998827f, -0.142509296562f, -0.04433270756f };

    std::vector<float> recurrentToForgetWeights = { -0.09499983487f, -0.08814888417f, -0.04834804721f, 0.1516668247f,
                                                    -0.3967529535f, -0.06463699788f, 0.4952811002f, 0.003274492938f,
                                                    -0.0968840941f, 0.17928104102f, 0.0031281141592f, -0.3387276584f,
                                                    -0.3587934076f, 0.06705895066f, 0.22463923692f, 0.1961955726f };

    std::vector<float> recurrentToCellWeights = { -0.21938985582f, -0.3023648226f, -0.1170005202f, -0.3509177422f,
                                                  -0.4286288613f, 0.2726137042f, 0.09216640889f, -0.06551410215f,
                                                  0.20453298098f, 0.2393476665f, 0.11846517771f, 0.2630801796f,
                                                  0.3954237699f, -0.19407111404f, 0.30412107706f, -0.27342408554f };

    std::vector<float> recurrentToOutputWeights = { -0.32921677827f, 0.32624614238f, -0.1388191282f, -0.17879831790f,
                                                    -0.15185534954f, -0.16918526583f, -0.10087361183f, -0.5436913968f,
                                                    0.016758225858f, 0.30454617738f, -0.41493862867f, -0.005565764375f,
                                                    -0.12584099173f, -0.12319286912f, 0.2407919466f, -0.08879069983f };
    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<float> cellToInputWeights;
    bool hasCellToForgetWeights = false;
    std::vector<float> cellToForgetWeights;
    bool hasCellToOutputWeights = false;
    std::vector<float> cellToOutputWeights;

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = {0., 0., 0., 0.};
    std::vector<float> forgetGateBias = {1., 1., 1., 1.};
    std::vector<float> cellBias = {0., 0., 0., 0.};
    std::vector<float> outputGateBias = {0., 0., 0., 0.};

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

    std::vector<float> inputValues = { 1., 2., 3., 4., 5., 4.,
                                       3., 2., 1., 2., 3., 4.,
                                       5., 4., 3., 2., 1., 2. };
    std::vector<float> expectedOutputValues = { -0.0714901f, -0.162117f, -0.175168f, -0.0232934f,
                                                -0.168107f, -0.414129f, -0.549875f, -0.00803579f,
                                                -0.0668735f, 0.204078f, -0.42765f, -0.0312321f,
                                                -0.120003f, -0.0941918f, -0.456391f, -0.0287019f,
                                                -0.0342921f, 0.20824f, -0.656989f, -0.00415265f,
                                                -0.10493f, 0.14211f, -0.583478f, -0.0329754f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<float>(backends,
                                              ::tflite::TensorType_FLOAT32,
                                              batchSize,
                                              timeSize,
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
                                              clippingThresProj,
                                              isTimeMajor);
}

void UnidirectionalSequenceLstmTimeMajorTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    std::vector<int32_t> inputShape = {timeSize, batchSize, inputSize};
    std::vector<int32_t> cellStateInTensorInfo = {batchSize, numUnits};
    std::vector<int32_t> outputStateInTensorInfo = {batchSize, outputSize};

    std::vector<int32_t> outputTensorInfo = {timeSize, batchSize, outputSize};

    //tensorInfo12
    bool hasInputToInputWeights = true;
    std::vector<float> inputToInputWeights = { 0.27277296781539917f, 0.3813590407371521f, -0.394489049911499f,
                                               0.2782636880874634f, -0.3793870210647583f, -0.018918335437774658f,
                                               0.2724653482437134f, -0.19314253330230713f, -0.2947450876235962f,
                                               -0.30253493785858154f, 0.4241350293159485f, -0.22560018301010132f };

    std::vector<float> inputToForgetWeights = { -0.2667974531650543f, -0.05505800247192383f, -0.20932340621948242f,
                                                -0.14345619082450867f, 0.09666192531585693f, -0.2604355812072754f,
                                                -0.2681812047958374f, -0.3314584493637085f, 0.4485899806022644f,
                                                -0.23467743396759033f, 0.5072842240333557f, -0.4192768931388855f };

    std::vector<float> inputToCellWeights = { -0.15782442688941956f, -0.027530014514923096f, 0.4789854884147644f,
                                              0.23227906227111816f, 0.28259342908859253f, -0.030095696449279785f,
                                              0.10071521997451782f, -0.08535495400428772f, 0.18563997745513916f,
                                              -0.3049069046974182f, -0.478048175573349f, 0.025234103202819824f };

    std::vector<float> inputToOutputWeights = { -0.04584759473800659f, -0.2716066539287567f, 0.012970447540283203f,
                                                -0.4729190170764923f, -0.37422770261764526f, 0.49352723360061646f,
                                                0.3163864016532898f, -0.436781644821167f, -0.33074596524238586f,
                                                -0.32885751128196716f, -0.40959352254867554f, -0.2124689817428589f };

    //tensorInfo16
    bool hasRecurrentToInputWeights = true;
    std::vector<float> recurrentToInputWeights = { 0.23788475990f, -0.24948765337f, 0.50044941902f, 0.14431896805f,
                                                   -0.115940228137f, -0.717082679f, -0.17208620906f, 0.17850610617f,
                                                   -0.16702319684f, -0.11384502053f, -0.309785276245f, -0.3316611672f,
                                                   0.52380162477f, -0.06839632987f, -0.391478359627f, -0.10756178963f };

    std::vector<float> recurrentToForgetWeights = { 0.11383482068f, 0.1676601767f, -0.08550968004f, 0.03399394089f,
                                                    0.08042152225f, -0.2133381964f, 0.05182432704f, 0.38161808255f,
                                                    -0.5018365979f, -0.08043262364f, 0.07894329014f, -0.07547105155f,
                                                    0.12047368288f, 0.2986997961f, 0.0485043078f, -0.13372567296f };

    std::vector<float> recurrentToCellWeights = { 0.0433832928545f, 0.07587072294f, -0.120520234107f, 0.604576051f,
                                                  -0.434353142986f, 0.009314475068f, 0.005085289478f, 0.08488202038f,
                                                  -0.00025437487886f, 0.15245915082f, -0.1936587542f, 0.004754020f,
                                                  -0.1582719236f, 0.3307867646f, 0.0236605107784f, 0.307716339826f };

    std::vector<float> recurrentToOutputWeights = { -0.079031050201f, 0.041414566286f, -0.583727357285f, 0.1025384515f,
                                                    -0.172372072937f, 0.09214124082f, 0.178184121827f, -0.2439443916f,
                                                    0.104485116899f, 0.2600405514f, 0.064414866268f, 0.24141204357f,
                                                    0.281875759363f, -0.14234502664f, 0.15126448862f, -0.24421440064f };
    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<float> cellToInputWeights;
    bool hasCellToForgetWeights = false;
    std::vector<float> cellToForgetWeights;
    bool hasCellToOutputWeights = false;
    std::vector<float> cellToOutputWeights;

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = {0., 0., 0., 0.};
    std::vector<float> forgetGateBias = {1., 1., 1., 1.};
    std::vector<float> cellBias = {0., 0., 0., 0.};
    std::vector<float> outputGateBias = {0., 0., 0., 0.};

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

    std::vector<float> inputValues = { 1., 2., 3., 4., 5., 4.,
                                       3., 2., 1., 2., 3., 4.,
                                       5., 4., 3., 2., 1., 2. };
    std::vector<float> expectedOutputValues = { 0.135658f, 0.124673f, 0.021209f, -0.0530204f,
                                                0.106138f, 0.0404792f, 0.0151644f, -0.00675166f,
                                                -0.0128514f, 0.0644884f, 0.0709072f, -0.0454045f,
                                                0.162886f, 0.166494f, 0.0277046f, -0.0369807f,
                                                0.111716f, 0.043119f, 0.0762981f, -0.0122854f,
                                                0.104397f, 0.2144f, 0.119192f, -0.0839058f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = true;

    UnidirectionalSequenceLstmTestImpl<float>(backends,
                                              ::tflite::TensorType_FLOAT32,
                                              batchSize,
                                              timeSize,
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
                                              clippingThresProj,
                                              isTimeMajor);
}

void UnidirectionalSequenceLstmNoCifgWithPeepholeWithProjectionTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 2;
    int32_t timeSize = 3;
    int32_t inputSize = 4;
    int32_t outputSize = 5;
    int32_t numUnits = 6;

    std::vector<int32_t> inputShape = {batchSize, timeSize, inputSize};
    std::vector<int32_t> cellStateInTensorInfo = {batchSize, numUnits};
    std::vector<int32_t> outputStateInTensorInfo = {batchSize, outputSize};

    std::vector<int32_t> outputTensorInfo = {batchSize, timeSize, outputSize};

    //tensorInfoInputSize,
    bool hasInputToInputWeights = true;
    std::vector<float> inputToInputWeights = { 0.021393683f, 0.06124551f, 0.046905167f, -0.014657677f,
                                               -0.03149463f, 0.09171803f, 0.14647801f, 0.10797193f,
                                               -0.0057968358f, 0.0019193048f, -0.2726754f, 0.10154029f,
                                               -0.018539885f, 0.080349885f, -0.10262385f, -0.022599787f,
                                               -0.09121155f, -0.008675967f, -0.045206103f, -0.0821282f,
                                               -0.008045952f, 0.015478081f, 0.055217247f, 0.038719587f };

    std::vector<float> inputToForgetWeights = { -0.0018401089f, -0.004852237f, 0.03698424f, 0.014181704f,
                                                0.028273236f, -0.016726194f, -0.05249759f, -0.10204261f,
                                                0.00861066f, -0.040979505f, -0.009899187f, 0.01923892f,
                                                -0.028177269f, -0.08535103f, -0.14585495f, 0.10662567f,
                                                -0.01909731f, -0.017883534f, -0.0047269356f, -0.045103323f,
                                                0.0030784295f, 0.076784775f, 0.07463696f, 0.094531395f};

    std::vector<float> inputToCellWeights = { -0.04580283f, -0.09549462f, -0.032418985f, -0.06454633f,
                                              -0.043528453f, 0.043018587f, -0.049152344f, -0.12418144f,
                                              -0.078985475f, -0.07596889f, 0.019484362f, -0.11434962f,
                                              -0.0074034138f, -0.06314844f, -0.092981495f, 0.0062155537f,
                                              -0.025034338f, -0.0028890965f, 0.048929527f, 0.06235075f,
                                              0.10665918f, -0.032036792f, -0.08505916f, -0.10843358f };

    std::vector<float> inputToOutputWeights = { -0.0998932f, -0.07201956f, -0.052803773f, -0.15629593f,
                                                -0.15001918f, -0.07650751f, 0.02359855f, -0.075155355f,
                                                -0.08037709f, -0.15093534f, 0.029517552f, -0.04751393f,
                                                0.010350531f, -0.02664851f, -0.016839722f, -0.023121163f,
                                                0.0077019283f, 0.012851257f, -0.05040649f, -0.0129761f,
                                                -0.021737747f, -0.038305793f, -0.06870586f, -0.01481247f };

    //tensorInfoOutputSize,
    bool hasRecurrentToInputWeights = true;
    std::vector<float> recurrentToInputWeights = { -0.001374326f, -0.078856036f, 0.10672688f, 0.029162422f,
                                                   -0.11585556f, 0.02557986f, -0.13446963f, -0.035785314f,
                                                   -0.01244275f, 0.025961924f, -0.02337298f, -0.044228926f,
                                                   -0.055839065f, -0.046598054f, -0.010546039f, -0.06900766f,
                                                   0.027239809f, 0.022582639f, -0.013296484f, -0.05459212f,
                                                   0.08981f, -0.045407712f, 0.08682226f, -0.06867011f,
                                                   -0.14390695f, -0.02916037f, 0.000996957f, 0.091420636f,
                                                   0.14283475f, -0.07390571f };

    std::vector<float> recurrentToForgetWeights = { -0.057784554f, -0.026057621f, -0.068447545f, -0.022581743f,
                                                   0.14811787f, 0.10826372f, 0.09471067f, 0.03987225f,
                                                   -0.0039523416f, 0.00030638507f, 0.053185795f, 0.10572994f,
                                                   0.08414449f, -0.022036452f, -0.00066928595f, -0.09203576f,
                                                   0.032950465f, -0.10985798f, -0.023809856f, 0.0021431844f,
                                                   -0.02196096f, -0.00326074f, 0.00058621005f, -0.074678116f,
                                                   -0.06193199f, 0.055729095f, 0.03736828f, 0.020123724f,
                                                   0.061878487f, -0.04729229f };

    std::vector<float> recurrentToCellWeights = { -0.037322544f, 0.018592842f, 0.0056175636f, -0.06253426f,
                                                   0.055647098f, -0.05713207f, -0.05626563f, 0.005559383f,
                                                   0.03375411f, -0.025757805f, -0.088049285f, 0.06017052f,
                                                   -0.06570978f, 0.007384076f, 0.035123326f, -0.07920549f,
                                                   0.053676967f, 0.044480428f, -0.07663568f, 0.0071805613f,
                                                   0.08089997f, 0.05143358f, 0.038261272f, 0.03339287f,
                                                   -0.027673481f, 0.044746667f, 0.028349208f, 0.020090483f,
                                                   -0.019443132f, -0.030755889f };

    std::vector<float> recurrentToOutputWeights = { 0.025825322f, -0.05813119f, 0.09495884f,
                                                    -0.045984812f,-0.01255415f, -0.0026479573f,
                                                    -0.08196161f, -0.054914974f, -0.0046604523f,
                                                    -0.029587349f, -0.044576716f, -0.07480124f,
                                                    -0.082868785f, 0.023254942f, 0.027502948f,
                                                    -0.0039728214f, -0.08683098f, -0.08116779f,
                                                    -0.014675607f, -0.037924774f, -0.023314456f,
                                                    -0.007401714f, -0.09255757f, 0.029460307f,
                                                    -0.08829125f, -0.005139627f, -0.08989442f,
                                                    -0.0555066f, 0.13596267f, 0.025062224f };
    // tensorInfoNumUnits
    bool hasCellToInputWeights = true;
    std::vector<float> cellToInputWeights = { 0.040369894f, 0.030746894f, 0.24704495f,
                                              0.018586371f, -0.037586458f, -0.15312155f };
    bool hasCellToForgetWeights = true;
    std::vector<float> cellToForgetWeights = { -0.01998659f, -0.15568835f, -0.24248174f,
                                               -0.012770197f, 0.041331276f, -0.072311886f };
    bool hasCellToOutputWeights = true;
    std::vector<float> cellToOutputWeights = { 0.08286371f, -0.08261836f, -0.51210177f,
                                               0.002913762f, 0.17764764f, -0.5495371f };

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = { 0.02234832f, 0.14757581f, 0.18176508f,
                                         0.10380666f, 0.053110216f, -0.06928846f };
    std::vector<float> forgetGateBias = { 0.035185695f, -0.042891346f, -0.03032477f,
                                          0.23027696f, 0.11098921f, 0.08989442f };
    std::vector<float> cellBias = { -0.024379363f, 0.0055531194f, 0.23377132f,
                                    0.033463873f, -0.1483596f, 0.029460307f };
    std::vector<float> outputGateBias =  { 0.046159424f, -0.0012809046f, 0.03563469f,
                                           0.12648113f, 0.027195795f, 0.35373217f };

    bool hasProjectionWeights = true;
    std::vector<float> projectionWeights = { -0.009802181f, 0.09401916f, 0.0717386f, -0.13895074f, 0.09641832f,
                                             0.060420845f, 0.08539281f, 0.054285463f, 0.061395317f, 0.034448683f,
                                             -0.042991187f, 0.019801661f, -0.16840284f, -0.015726732f, -0.23041931f,
                                             -0.024478018f, -0.10959692f, -0.013875541f, 0.18600968f, -0.061274476f,
                                             0.0138165f, -0.08160894f, -0.07661644f, 0.032372914f, 0.16169067f,
                                             0.22465782f, -0.03993472f, -0.004017731f, 0.08633481f, -0.28869787f };

    bool hasProjectionBias = true;
    std::vector<float> projectionBias(outputSize, 0.f);

    bool hasInputLayerNormWeights = false;
    std::vector<float> inputLayerNormWeights;
    bool hasForgetLayerNormWeights = false;
    std::vector<float> forgetLayerNormWeights;
    bool hasCellLayerNormWeights = false;
    std::vector<float> cellLayerNormWeights;
    bool hasOutputLayerNormWeights = false;
    std::vector<float> outputLayerNormWeights;

    std::vector<float> inputValues = { 1., 2., 3., 4., 5., 4.,
                                       3., 2., 1., 2., 3., 4.,
                                       5., 4., 3., 2., 1., 2.,
                                       1., 2., 3., 4., 5., 4.};
    std::vector<float> expectedOutputValues = { -0.0135612f, -0.0263441f, 0.0314008f, -0.00883455f, 0.00763052f,
                                                -0.00126877f, -0.0292959f, 0.0449957f, -0.00976195f, -0.00492338f,
                                                -0.0175702f, -0.0431753f, 0.0597117f, -0.0169154f, 0.0142087f,
                                                0.00472515f, -0.0196355f, 0.0342524f, -0.00407936f, -0.0253189f,
                                                -0.00512944f, -0.0293754f, 0.0512771f, -0.0151874f, -0.0246433f,
                                                -0.00744986f, -0.0345103f, 0.0450666f, -0.00944991f, 0.0126895f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<float>(backends,
                                              ::tflite::TensorType_FLOAT32,
                                              batchSize,
                                              timeSize,
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
                                              clippingThresProj,
                                              isTimeMajor);
}

void UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    //tensorInfo12
    bool hasInputToInputWeights = false;
    std::vector<float> inputToInputWeights{};

    std::vector<float> inputToForgetWeights = { 0.2415594226f, 0.15400093799f, 0.4566498398f,
                                                -0.3810434485f, 0.268383264f, -0.009807467424f,
                                                -0.3522925403f, -0.24275735512f, -0.28344226125f,
                                                0.13512269116f, -0.4932442977f, -0.10039821991f };

    std::vector<float> inputToCellWeights = { -0.2504855627f, 0.184490025045f, -0.2480507493f,
                                              0.386399507f, -0.259465157985f, -0.16545993089f,
                                              -0.4230232555f, 0.341664791103f, -0.18127849691f,
                                              -0.2277662414f, -0.55275535589f, 0.34184026718f };

    std::vector<float> inputToOutputWeights = { 0.2303854227f, 0.5218806862f, -0.4865379333f,
                                                0.53969591851f, 0.23393625035f, -0.27140527306f,
                                                0.50009280443f, 0.07511717046f, 0.3998299249f,
                                                -0.51717478049f, 0.1889653282f, -0.367323637f };

    //tensorInfo16
    bool hasRecurrentToInputWeights = false;
    std::vector<float> recurrentToInputWeights{};

    std::vector<float> recurrentToForgetWeights = { -0.09499983487f, -0.08814888417f, -0.04834804721f, 0.1516668247f,
                                                    -0.3967529535f, -0.06463699788f, 0.4952811002f, 0.003274492938f,
                                                    -0.0968840941f, 0.17928104102f, 0.0031281141592f, -0.3387276584f,
                                                    -0.3587934076f, 0.06705895066f, 0.22463923692f, 0.1961955726f };

    std::vector<float> recurrentToCellWeights = { -0.21938985582f, -0.3023648226f, -0.1170005202f, -0.3509177422f,
                                                  -0.4286288613f, 0.2726137042f, 0.09216640889f, -0.06551410215f,
                                                  0.20453298098f, 0.2393476665f, 0.11846517771f, 0.2630801796f,
                                                  0.3954237699f, -0.19407111404f, 0.30412107706f, -0.27342408554f };

    std::vector<float> recurrentToOutputWeights = { -0.32921677827f, 0.32624614238f, -0.1388191282f, -0.17879831790f,
                                                    -0.15185534954f, -0.16918526583f, -0.10087361183f, -0.5436913968f,
                                                    0.016758225858f, 0.30454617738f, -0.41493862867f, -0.005565764375f,
                                                    -0.12584099173f, -0.12319286912f, 0.2407919466f, -0.08879069983f };
    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<float> cellToInputWeights;
    bool hasCellToForgetWeights = true;
    std::vector<float> cellToForgetWeights =  {0.47485286f, -0.51955009f, -0.24458408f, 0.31544167f};
    bool hasCellToOutputWeights = true;
    std::vector<float> cellToOutputWeights =  {-0.17135078f, 0.82760304f, 0.85573703f, -0.77109635f};

    bool hasInputGateBias = false;
    std::vector<float> inputGateBias;
    std::vector<float> forgetGateBias = {1., 1., 1., 1.};
    std::vector<float> cellBias =  {0., 0., 0., 0.};
    std::vector<float> outputGateBias = {0., 0., 0., 0.};

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

    std::vector<float> inputValues = { 1., 2., 3., 4., 5., 4.,
                                       3., 2., 1., 2., 3., 4.,
                                       5., 4., 3., 2., 1., 2. };
    std::vector<float> expectedOutputValues =  { -0.0129257f, -0.070531f, -0.153508f, -0.0392391f,
                                                 -0.0300169f, -0.195717f, -0.528679f, -0.0818106f,
                                                 -0.0332748f, 0.155429f, -0.353966f, -0.0801505f,
                                                 -0.032312f, -0.0407911f, -0.435053f, -0.0932317f,
                                                 -0.0108233f, 0.165584f, -0.640424f, -0.0447535f,
                                                 -0.031675f, 0.125987f, -0.526695f, -0.110093f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<float>(backends,
                                              ::tflite::TensorType_FLOAT32,
                                              batchSize,
                                              timeSize,
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
                                              clippingThresProj,
                                              isTimeMajor);
}

void UnidirectionalSequenceLstmNoCifgWithPeepholeWithProjectionWithLayerNormTest(
    std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    int32_t numUnits = 5;

    //tensorInfo15
    bool hasInputToInputWeights = true;
    std::vector<float> inputToInputWeights = { -0.49536117f, -0.0556083915f, -0.102400711f,
                                               -0.117484632f, 0.3298470976f, -0.1179017122f,
                                               0.214305695f, 0.42135173085f, 0.003878414626f,
                                               -0.348303917f, -0.1881275477f, 0.0343011027f,
                                               -0.38837709614f, -0.05636804124f, 0.4259087456f};

    std::vector<float> inputToForgetWeights = { 0.2415594226f, 0.15400093799f, 0.4566498398f,
                                                -0.3810434485f, 0.268383264f, -0.009807467424f,
                                                -0.3522925403f, -0.24275735512f, -0.28344226125f,
                                                0.13512269116f, -0.4932442977f, -0.10039821991f,
                                                0.2726137042f, 0.09216640889f, -0.06551410215f};

    std::vector<float> inputToCellWeights = { -0.2504855627f, 0.184490025045f, -0.2480507493f,
                                              0.386399507f, -0.259465157985f, -0.16545993089f,
                                              -0.4230232555f, 0.341664791103f, -0.18127849691f,
                                              -0.2277662414f, -0.55275535589f, 0.34184026718f,
                                              0.3954237699f, -0.19407111404f, 0.30412107706f};

    std::vector<float> inputToOutputWeights = { 0.2303854227f, 0.5218806862f, -0.4865379333f,
                                                0.53969591851f, 0.23393625035f, -0.27140527306f,
                                                0.50009280443f, 0.07511717046f, 0.3998299249f,
                                                -0.51717478049f, 0.1889653282f, -0.367323637f,
                                                -0.12584099173f, -0.12319286912f, 0.2407919466f};

    //tensorInfo20
    bool hasRecurrentToInputWeights = true;
    std::vector<float> recurrentToInputWeights = { -0.128009796112f, 0.1995525098f, -0.07745539397f, 0.1558421701f,
                                                   -0.265254765766f, -0.38837709614f, -0.05636804124f, 0.4259087456f,
                                                   0.17628988623f, 0.3877420127f, 0.53300309181f, -0.0959980934f,
                                                   0.00302857416f, 0.3266998827f, -0.142509296562f, -0.04433270756f,
                                                   0.54066205f, -0.32668582f, -0.43562764f, -0.56094903f };

    std::vector<float> recurrentToForgetWeights = { -0.09499983487f, -0.08814888417f, -0.04834804721f, 0.1516668247f,
                                                    -0.3967529535f, -0.06463699788f, 0.4952811002f, 0.003274492938f,
                                                    -0.0968840941f, 0.17928104102f, 0.0031281141592f, -0.3387276584f,
                                                    -0.3587934076f, 0.06705895066f, 0.22463923692f, 0.1961955726f,
                                                    0.01841056f, -0.32764608f, -0.33027974f, -0.10826075f };

    std::vector<float> recurrentToCellWeights = { -0.21938985582f, -0.3023648226f, -0.1170005202f, -0.3509177422f,
                                                  -0.4286288613f, 0.2726137042f, 0.09216640889f, -0.06551410215f,
                                                  0.20453298098f, 0.2393476665f, 0.11846517771f, 0.2630801796f,
                                                  0.3954237699f, -0.19407111404f, 0.30412107706f, -0.27342408554f,
                                                  0.19069612f, -0.03026325f, -0.54532051f, 0.33003211f };

    std::vector<float> recurrentToOutputWeights = { -0.32921677827f, 0.32624614238f, -0.1388191282f, -0.17879831790f,
                                                    -0.15185534954f, -0.16918526583f, -0.10087361183f, -0.5436913968f,
                                                    0.016758225858f, 0.30454617738f, -0.41493862867f, -0.005565764375f,
                                                    -0.12584099173f, -0.12319286912f, 0.2407919466f, -0.08879069983f,
                                                    0.11178309f, 0.09481031f, -0.26424935f, 0.46261835f };
    // tensorInfo5
    bool hasCellToInputWeights = true;
    std::vector<float> cellToInputWeights = { 0.05f, 0.1f, 0.25f, 0.15f, -0.02f };
    bool hasCellToForgetWeights = true;
    std::vector<float> cellToForgetWeights = { -0.02f, -0.15f, -0.25f, -0.03f, 0.15f };
    bool hasCellToOutputWeights = true;
    std::vector<float> cellToOutputWeights = { 0.1f, -0.1f, -0.5f, 0.05f, 0.01f };

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = { 0.03f, 0.15f, 0.22f, 0.38f, 0.05f };
    std::vector<float> forgetGateBias = { 0.1f, -0.3f, -0.2f, 0.1f, 0.4f };
    std::vector<float> cellBias = { -0.05f, 0.72f, 0.25f, 0.08f, 0.1f };
    std::vector<float> outputGateBias = { 0.05f, -0.01f, 0.2f, 0.1f, -0.2f };

    bool hasProjectionWeights = true;
    std::vector<float> projectionWeights = { -0.1f, 0.2f, 0.01f, -0.2f,
                                             0.1f, 0.5f,  0.3f, 0.08f,
                                             0.07f, 0.2f, -0.4f,  0.2f,
                                             0.5f, -0.4f, 0.3f, -0.2f,
                                             0.3f, 0.08f, -0.07f, 0.2f}; //{outputSize, numUnits}
    bool hasProjectionBias = true;
    std::vector<float> projectionBias(outputSize, 0.f);;

    bool hasInputLayerNormWeights = true;
    std::vector<float> inputLayerNormWeights = { 0.1f, 0.2f, 0.3f, 0.5f, 0.8f };
    bool hasForgetLayerNormWeights = true;
    std::vector<float> forgetLayerNormWeights = { 0.1f, 0.2f, 0.3f, 0.5f, 0.2f };
    bool hasCellLayerNormWeights = true;
    std::vector<float> cellLayerNormWeights = { 0.7f, 0.2f, 0.3f, 0.8f, 0.5f };
    bool hasOutputLayerNormWeights = true;
    std::vector<float> outputLayerNormWeights = { 0.6f, 0.2f, 0.2f, 0.5f, 0.1f };

    std::vector<float> inputValues = { 1., 2., 3., 4., 5., 4.,
                                       3., 2., 1., 2., 3., 4.,
                                       5., 4., 3., 2., 1., 2. };
    std::vector<float> expectedOutputValues = { 0.0642256f, 0.0343966f, 0.184122f, 0.114717f,
                                                0.11458f, 0.0407109f, 0.300327f, 0.174301f,
                                                0.0864761f, 0.0362912f, 0.178635f, 0.115689f,
                                                0.108008f, 0.0386623f, 0.273471f, 0.167115f,
                                                0.0859545f, 0.0331481f, 0.186051f, 0.11888f,
                                                0.106649f, 0.0276847f, 0.229863f, 0.166958f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<float>(backends,
                                              ::tflite::TensorType_FLOAT32,
                                              batchSize,
                                              timeSize,
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
                                              clippingThresProj,
                                              isTimeMajor);
}

void UnidirectionalSequenceLstmInt8Test(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    //tensorInfo12
    bool hasInputToInputWeights = true;
    std::vector<int8_t> inputToInputWeights = { -4, -1, -1, -2, 3, -2, 2, 4, 1, -4, -2, 3 };

    std::vector<int8_t> inputToForgetWeights = { 2, 1, 4, -4, 3, -1, -3, -2, -3, 1, -4, -1 };

    std::vector<int8_t> inputToCellWeights = { -2, 1, -2, 4, -3, -2, -4, 3, -2, -2, -6, 3 };

    std::vector<int8_t> inputToOutputWeights = { 2, 5, -4, 5, 2, -3, 5, 7, 3, -5, 1, -4 };

    //tensorInfo16
    bool hasRecurrentToInputWeights = true;
    std::vector<int8_t> recurrentToInputWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -1, -1 };

    std::vector<int8_t> recurrentToForgetWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -2, -1 };

    std::vector<int8_t> recurrentToCellWeights = { -2, -3, -1, -3, -4, 2, 1, -1, 2, 2, 1, 2, 3, -2, 3, -3 };

    std::vector<int8_t> recurrentToOutputWeights = { -3, 3, -1, -2, -2, -2, -1, -5, 1, 3, -4, -1, -1, -1, 2, -1 };

    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<int8_t> cellToInputWeights;
    bool hasCellToForgetWeights = false;
    std::vector<int8_t> cellToForgetWeights;
    bool hasCellToOutputWeights = false;
    std::vector<int8_t> cellToOutputWeights;

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = { 0., 0., 0., 0. };
    std::vector<float> forgetGateBias = { 1., 1., 1., 1. };
    std::vector<float> cellBias = { 0., 0., 0., 0. };
    std::vector<float> outputGateBias = { 0., 0., 0., 0. };

    bool hasProjectionWeights = false;
    std::vector<int8_t> projectionWeights;
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

    std::vector<float> inputValues = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f,
                                       0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f,
                                       0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f };

    std::vector<float> expectedOutputValues = { -0.0142517f, -0.0198845f, -0.0120569f, -0.0116868f,
                                                -0.0350714f, -0.0343202f, -0.047504f, -0.0569789f,
                                                -0.0146346f, 0.0106663f, -0.0247238f, -0.0319502f,
                                                -0.0294759f, -0.0129935f, -0.0444175f, -0.0444354f,
                                                -0.0280855f, 0.00545101f, -0.051422f, -0.0463838f,
                                                -0.0310702f, 0.00915739f, -0.0625207f, -0.0482648f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<int8_t>(backends,
                                               ::tflite::TensorType_INT8,
                                               batchSize,
                                               timeSize,
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
                                               clippingThresProj,
                                               isTimeMajor,
                                               0.1f);
}

void UnidirectionalSequenceLstmInt8TimeMajorTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    //tensorInfo12
    bool hasInputToInputWeights = true;
    std::vector<int8_t> inputToInputWeights = { -4, -1, -1, -2, 3, -2, 2, 4, 1, -4, -2, 3 };

    std::vector<int8_t> inputToForgetWeights = { 2, 1, 4, -4, 3, -1, -3, -2, -3, 1, -4, -1 };

    std::vector<int8_t> inputToCellWeights = { -2, 1, -2, 4, -3, -2, -4, 3, -2, -2, -6, 3 };

    std::vector<int8_t> inputToOutputWeights = { 2, 5, -4, 5, 2, -3, 5, 7, 3, -5, 1, -4 };

    //tensorInfo16
    bool hasRecurrentToInputWeights = true;
    std::vector<int8_t> recurrentToInputWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -1, -1 };

    std::vector<int8_t> recurrentToForgetWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -2, -1 };

    std::vector<int8_t> recurrentToCellWeights = { -2, -3, -1, -3, -4, 2, 1, -1, 2, 2, 1, 2, 3, -2, 3, -3 };

    std::vector<int8_t> recurrentToOutputWeights = { -3, 3, -1, -2, -2, -2, -1, -5, 1, 3, -4, -1, -1, -1, 2, -1 };

    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<int8_t> cellToInputWeights;
    bool hasCellToForgetWeights = false;
    std::vector<int8_t> cellToForgetWeights;
    bool hasCellToOutputWeights = false;
    std::vector<int8_t> cellToOutputWeights;

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = { 0., 0., 0., 0. };
    std::vector<float> forgetGateBias = { 1., 1., 1., 1. };
    std::vector<float> cellBias = { 0., 0., 0., 0. };
    std::vector<float> outputGateBias = { 0., 0., 0., 0. };

    bool hasProjectionWeights = false;
    std::vector<int8_t> projectionWeights;
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

    std::vector<float> inputValues = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f,
                                       0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f,
                                       0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f };

    std::vector<float> expectedOutputValues = { -0.0142517f, -0.0198845f, -0.0120122f, -0.0116868f,
                                                -0.0261295f, -0.0188487f, -0.0345463f, -0.049733f,
                                                -0.0146346f, 0.0106663f, -0.0247238f, -0.0319502f,
                                                -0.0291863f, -0.0369402f, -0.0354071f, -0.0296529f,
                                                -0.0419539f, -0.00617731f, -0.0814796f, -0.0804005f,
                                                -0.0244737f, 0.0119905f, -0.0457527f, -0.0331862f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = true;

    UnidirectionalSequenceLstmTestImpl<int8_t>(backends,
                                               ::tflite::TensorType_INT8,
                                               batchSize,
                                               timeSize,
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
                                               clippingThresProj,
                                               isTimeMajor,
                                               0.1);
}

void UnidirectionalSequenceLstmInt8NoCifgWithPeepholeWithProjectionTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    int32_t numUnits = 4;

    bool hasInputToInputWeights = true;
    std::vector<int8_t> inputToInputWeights = { -4, -1, -1, -2, 3, -2, 2, 4, 1, -4, -2, 3 };

    std::vector<int8_t> inputToForgetWeights = { 2, 1, 4, -4, 3, -1, -3, -2, -3, 1, -4, -1 };

    std::vector<int8_t> inputToCellWeights = { -2, 1, -2, 4, -3, -2, -4, 3, -2, -2, -6, 3 };

    std::vector<int8_t> inputToOutputWeights = { 2, 5, -4, 5, 2, -3, 5, 7, 3, -5, 1, -4 };

    //tensorInfo16
    bool hasRecurrentToInputWeights = true;
    std::vector<int8_t> recurrentToInputWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -1, -1 };

    std::vector<int8_t> recurrentToForgetWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -2, -1 };

    std::vector<int8_t> recurrentToCellWeights = { -2, -3, -1, -3, -4, 2, 1, -1, 2, 2, 1, 2, 3, -2, 3, -3 };

    std::vector<int8_t> recurrentToOutputWeights = { -3, 3, -1, -2, -2, -2, -1, -5, 1, 3, -4, -1, -1, -1, 2, -1 };

    // tensorInfo4
    bool hasCellToInputWeights = true;
    std::vector<int8_t> cellToInputWeights = { 5, 10, 25, 15 };
    bool hasCellToForgetWeights = true;
    std::vector<int8_t> cellToForgetWeights = { -5, 15, 25, 3 };
    bool hasCellToOutputWeights = true;
    std::vector<int8_t> cellToOutputWeights = { 10, -10, -5, 50 };

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = { 0.02234832f,  0.14757581f,   0.18176508f,  0.10380666f};
    std::vector<float> forgetGateBias = { 0.035185695f, -0.042891346f, -0.3032477f, 0.23027696f};
    std::vector<float> cellBias = { -0.124379363f, 0.55531194f, 0.23377132f,   0.033463873f };
    std::vector<float> outputGateBias = { 0.046159424f,  -0.12809046f, 0.03563469f, 0.12648113f };

    bool hasProjectionWeights = true;
    std::vector<int8_t> projectionWeights = { -25, 51, 3, -5, 25, 127, 77, 20, 18, 51, -10, 51, -25, 88, 77, -13 };
    bool hasProjectionBias = true;
    std::vector<float> projectionBias(outputSize, 0.f);

    bool hasInputLayerNormWeights = false;
    std::vector<float> inputLayerNormWeights;
    bool hasForgetLayerNormWeights = false;
    std::vector<float> forgetLayerNormWeights;
    bool hasCellLayerNormWeights = false;
    std::vector<float> cellLayerNormWeights;
    bool hasOutputLayerNormWeights = false;
    std::vector<float> outputLayerNormWeights;

    std::vector<float> inputValues = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f,
                                       0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f,
                                       0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f };

    std::vector<float> expectedOutputValues = { 0.612103f, 1.56788f, 0.31966f, 1.42956f,
                                                0.909718f, 3.07916f, -0.560586f, 3.8907f,
                                                0.753671f, 1.77485f, 0.365122f, 1.60077f,
                                                0.812644f, 2.79092f, -0.605396f, 3.61742f,
                                                0.791857f, 1.64353f, 0.316588f, 1.55192f,
                                                0.807265f, 2.47012f, -0.539598f, 3.25654f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<int8_t>(backends,
                                               ::tflite::TensorType_INT8,
                                               batchSize,
                                               timeSize,
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
                                               clippingThresProj,
                                               isTimeMajor,
                                               0.1f);
}

void UnidirectionalSequenceLstmInt8WithCifgWithPeepholeNoProjectionTest(std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    // cellSize and outputSize have the same size when there is no projection.
    int32_t numUnits = outputSize;

    //tensorInfo12,
    bool hasInputToInputWeights = false;
    std::vector<int8_t> inputToInputWeights;

    std::vector<int8_t> inputToForgetWeights = { 2, 1, 4, -4, 3, -1, -3, -2, -3, 1, -4, -1 };

    std::vector<int8_t> inputToCellWeights = { -2, 1, -2, 4, -3, -2, -4, 3, -2, -2, -6, 3 };

    std::vector<int8_t> inputToOutputWeights = { 2, 5, -4, 5, 2, -3, 5, 7, 3, -5, 1, -4 };

    //tensorInfo16,
    bool hasRecurrentToInputWeights = false;
    std::vector<int8_t> recurrentToInputWeights;
    std::vector<int8_t> recurrentToForgetWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3, 5, -1, 1, 3, -2, -1 };

    std::vector<int8_t> recurrentToCellWeights = { -2, -3, -1, -3, -4, 2, 1, -1, 2, 2, 1, 2, 3, -2, 3, -3 };

    std::vector<int8_t> recurrentToOutputWeights = { -3, 3, -1, -2, -2, -2, -1, -5, 1, 3, -4, -1, -1, -1, 2, -1 };

    // tensorInfo4
    bool hasCellToInputWeights = false;
    std::vector<int8_t> cellToInputWeights;
    bool hasCellToForgetWeights = true;
    std::vector<int8_t> cellToForgetWeights = { 47, -52, -24, 31 };
    bool hasCellToOutputWeights = true;
    std::vector<int8_t> cellToOutputWeights = { -17, 82, 85, -77 };

    bool hasInputGateBias = false;
    std::vector<float> inputGateBias;
    std::vector<float> forgetGateBias = { 1., 1., 1., 1. };
    std::vector<float> cellBias = { 0., 0., 0., 0. };
    std::vector<float> outputGateBias = { 0., 0., 0., 0. };

   bool hasProjectionWeights = false;
    std::vector<int8_t> projectionWeights;
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

    std::vector<float> inputValues = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f,
                                       0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f,
                                       0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f };

    std::vector<float> expectedOutputValues = { -0.0072104f, -0.00991171f, -0.00650478f, -0.00713055f,
                                                -0.0191782f, -0.0161269f, -0.0233683f, -0.054299f,
                                                -0.00783725f, 0.00635271f, -0.0126718f, -0.022613f,
                                                -0.0161351f, -0.00775868f, -0.021054f, -0.0339778f,
                                                -0.0146392f, 0.00330261f, -0.0258733f, -0.0407797f,
                                                -0.0174297f, 0.0050105f, -0.0266275f, -0.0362564f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<int8_t>(backends,
                                               ::tflite::TensorType_INT8,
                                               batchSize,
                                               timeSize,
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
                                               clippingThresProj,
                                               isTimeMajor,
                                               0.1);
}

void UnidirectionalSequenceLstmInt8NoCifgWithPeepholeWithProjectionWithLayerNormTest(
    std::vector<armnn::BackendId>& backends)
{
    int32_t batchSize = 3;
    int32_t timeSize = 2;
    int32_t inputSize = 3;
    int32_t outputSize = 4;
    int32_t numUnits = 5;

    bool hasInputToInputWeights = true;
    std::vector<int8_t> inputToInputWeights = { -4, -1, -1, -2, 3, -2, 2, 4, 1, -4, -2, 3, 2, 2, -4 };

    std::vector<int8_t> inputToForgetWeights = { 2, 1, 4, -4, 3, -1, -3, -2, -3, 1, -4, -1, -3, -2, -4 };

    std::vector<int8_t> inputToCellWeights = { -2, 1, -2, 4, -3, -2, -4, 3, -2, -2, -6, 3, 2, 5, -4 };

    std::vector<int8_t> inputToOutputWeights = { 2, 5, -4, 5, 2, -3, 5, 7, 3, -5, 1, -4, -4, -1, -1 };

    bool hasRecurrentToInputWeights = true;
    std::vector<int8_t> recurrentToInputWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3,
                                                    5, -1, 1, 3, -1, -1, -1, 4, 2, 3 };

    std::vector<int8_t> recurrentToForgetWeights = { -1, 1, -1, 1, -3, -4, -1, 4, 2, 3,
                                                     5, -1, 1, 3, -2, -1, -1, 2, 2, 1 };

    std::vector<int8_t> recurrentToCellWeights = { -2, -3, -1, -3, -4, 2, 1, -1, 2, 2,
                                                   1, 2, 3, -2, 3, -3,  -1, -5, 1, 3 };

    std::vector<int8_t> recurrentToOutputWeights = { -3, 3, -1, -2, -2, -2, -1, -5, 1, 3,
                                                     -4, -1, -1, -1, 2, -1, 5, 1, -3, -4 };

    // tensorInfo5
    bool hasCellToInputWeights = true;
    std::vector<int8_t> cellToInputWeights = { 5, 3, 8, -5, 2 };
    bool hasCellToForgetWeights = true;
    std::vector<int8_t> cellToForgetWeights = { -2, -7, 5, -3, 4 };
    bool hasCellToOutputWeights = true;
    std::vector<int8_t> cellToOutputWeights = { 9, -10 , -5, 5, 1 };

    bool hasInputGateBias = true;
    std::vector<float> inputGateBias = { 0.03f, 0.15f, 0.22f, 0.38f, 0.05f };
    std::vector<float> forgetGateBias = { 0.1f, -0.3f, -0.2f, 0.1f, 0.4f };
    std::vector<float> cellBias = { -0.05f, 0.72f, 0.25f, 0.08f, 0.1f };
    std::vector<float> outputGateBias = { 0.05f, -0.01f, 0.2f, 0.1f, -0.2f };

    bool hasProjectionWeights = true;
    std::vector<int8_t> projectionWeights = { -1, 2, 1, -2, 1, 5, 3, 8, 7, 2,
                                              -4, 2, 5, -4, 3, -2, 3, 8, -7, 2 };
    bool hasProjectionBias = true;
    std::vector<float> projectionBias(outputSize, 0.f);

    bool hasInputLayerNormWeights = true;
    std::vector<float> inputLayerNormWeights = { 0.1f, 0.2f, -0.3f, -0.1f, 0.5f };
    bool hasForgetLayerNormWeights = true;
    std::vector<float> forgetLayerNormWeights = { -0.1f, 0.2f, 0.3f, 0.5f, 0.2f };
    bool hasCellLayerNormWeights = true;
    std::vector<float> cellLayerNormWeights = { 0.5f, 0.2f, 0.3f, 0.4f, -0.5f };
    bool hasOutputLayerNormWeights = true;
    std::vector<float> outputLayerNormWeights = { 0.6f, -0.2f, -0.2f, 0.5f, 0.1f };

    std::vector<float> inputValues = { 1., 8., 3., 4., 5., 4.,
                                       3., 2., 1., 2., 3., 4.,
                                       5., 4., 3., 2., 1., 2. };

    std::vector<float> expectedOutputValues = { 0.0471276f, 0.0168155f, 0.0789885f, 0.16550f,
                                                0.0643133f, -0.0400722f, 0.100593f, 0.197722f,
                                                0.0465562f, -0.0600682f, 0.0622087f, 0.115053f,
                                                0.056287f, -0.0566218f, 0.0856832f, 0.148484f,
                                                0.0457859f, -0.0588112f, 0.0623636f, 0.114333f,
                                                0.0509271f, -0.0754262f, 0.058600f, 0.0801288f };

    tflite::ActivationFunctionType activationFunction = tflite::ActivationFunctionType_TANH;
    float clippingThresCell = 10.f;
    float clippingThresProj = 0.f;
    bool isTimeMajor = false;

    UnidirectionalSequenceLstmTestImpl<int8_t>(backends,
                                               ::tflite::TensorType_INT8,
                                               batchSize,
                                               timeSize,
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
                                               clippingThresProj,
                                               isTimeMajor,
                                               0.1);
}

TEST_SUITE("UnidirectionalSequenceLstmTest_CpuRefTests")
{

TEST_CASE ("UnidirectionalSequenceLstmTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmTimeMajorTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmTimeMajorTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmNoCifgWithPeepholeWithProjectionTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmNoCifgWithPeepholeWithProjectionTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmNoCifgWithPeepholeWithProjectionWithLayerNormTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmNoCifgWithPeepholeWithProjectionWithLayerNormTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmInt8Test_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmInt8Test(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmTimeInt8TimeMajorTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmInt8TimeMajorTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmInt8NoCifgWithPeepholeWithProjectionTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmInt8NoCifgWithPeepholeWithProjectionTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmInt8WithCifgWithPeepholeNoProjectionTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmInt8WithCifgWithPeepholeNoProjectionTest(backends);
}

TEST_CASE ("UnidirectionalSequenceLstmInt8NoCifgWithPeepholeWithProjectionWithLayerNormTest_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    UnidirectionalSequenceLstmInt8NoCifgWithPeepholeWithProjectionWithLayerNormTest(backends);
}

} //End of TEST_SUITE("UnidirectionalSequenceLstmTest_CpuRef")

} // namespace armnnDelegate