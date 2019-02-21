//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../TfLiteParser.hpp"

#include <boost/test/unit_test.hpp>
#include "test/GraphUtils.hpp"

#include "ParserFlatbuffersFixture.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct DetectionPostProcessFixture : ParserFlatbuffersFixture
{
    explicit DetectionPostProcessFixture()
    {
        /*
            The following values were used for the custom_options:
            use_regular_non_max_suppression = true
            max_classes_per_detection = 1
            nms_score_threshold = 0.0
            nms_iou_threshold = 0.5
            max_detections = 3
            max_detections = 3
            num_classes = 2
            h_scale = 5
            w_scale = 5
            x_scale = 10
            y_scale = 10
        */
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [{
                    "builtin_code": "CUSTOM",
                    "custom_code": "TFLite_Detection_PostProcess"
                }],
                "subgraphs": [{
                    "tensors": [{
                            "shape": [1, 6, 4],
                            "type": "UINT8",
                            "buffer": 0,
                            "name": "box_encodings",
                            "quantization": {
                                "min": [0.0],
                                "max": [255.0],
                                "scale": [1.0],
                                "zero_point": [ 1 ]
                            }
                        },
                        {
                            "shape": [1, 6, 3],
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "scores",
                            "quantization": {
                                "min": [0.0],
                                "max": [255.0],
                                "scale": [0.01],
                                "zero_point": [0]
                            }
                        },
                        {
                            "shape": [6, 4],
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "anchors",
                            "quantization": {
                                "min": [0.0],
                                "max": [255.0],
                                "scale": [0.5],
                                "zero_point": [0]
                            }
                        },
                        {
                            "shape": [1, 3, 4],
                            "type": "FLOAT32",
                            "buffer": 3,
                            "name": "detection_boxes",
                            "quantization": {}
                        },
                        {
                            "shape": [1, 3],
                            "type": "FLOAT32",
                            "buffer": 4,
                            "name": "detection_classes",
                            "quantization": {}
                        },
                        {
                            "shape": [1, 3],
                            "type": "FLOAT32",
                            "buffer": 5,
                            "name": "detection_scores",
                            "quantization": {}
                        },
                        {
                            "shape": [1],
                            "type": "FLOAT32",
                            "buffer": 6,
                            "name": "num_detections",
                            "quantization": {}
                        }
                    ],
                    "inputs": [0, 1, 2],
                    "outputs": [3, 4, 5, 6],
                    "operators": [{
                        "opcode_index": 0,
                        "inputs": [0, 1, 2],
                        "outputs": [3, 4, 5, 6],
                        "builtin_options_type": 0,
                        "custom_options": [
                            109, 97, 120, 95, 100, 101, 116, 101, 99, 116, 105, 111, 110, 115, 0, 109, 97,
                            120, 95, 99, 108, 97, 115, 115, 101, 115, 95, 112, 101, 114, 95, 100, 101, 116,
                            101, 99,  116,  105, 111, 110, 0, 110, 109, 115, 95, 115, 99, 111, 114, 101, 95,
                            116, 104, 114, 101, 115, 104, 111, 108, 100, 0, 110, 109, 115, 95, 105, 111, 117,
                            95, 116, 104, 114, 101, 115, 104, 111, 108, 100, 0, 110, 117, 109, 95, 99, 108, 97,
                            115, 115, 101, 115, 0, 104, 95, 115, 99, 97, 108, 101, 0, 119, 95, 115, 99, 97,
                            108, 101, 0, 120, 95, 115, 99, 97, 108, 101, 0, 121, 95, 115, 99, 97, 108, 101, 0,
                            117, 115, 101, 95, 114, 101, 103, 117, 108, 97, 114, 95, 110, 111, 110, 95, 109, 97,
                            120, 95, 115, 117, 112, 112, 114, 101, 115, 115, 105, 111, 110, 0, 100, 101, 116,
                            101, 99, 116, 105, 111, 110, 115, 95, 112, 101, 114, 95, 99, 108, 97, 115, 115, 0,
                            11, 22, 87, 164, 180, 120, 141, 104, 61, 86, 79, 72, 11, 0, 0, 0, 1, 0, 0, 0, 11, 0,
                            0, 0, 1, 0, 0, 0, 0, 0, 160, 64, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 2,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 160, 64, 0, 0, 32, 65, 0, 0, 32, 65, 6, 14, 6, 6, 14, 14,
                            6, 106, 14, 14, 14, 55, 38, 1
                        ],
                        "custom_options_format": "FLEXBUFFERS"
                    }]
                }],
                "buffers": [{},
                    {},
                    { "data": [ 1, 1,   2, 2,
                                1, 1,   2, 2,
                                1, 1,   2, 2,
                                1, 21,  2, 2,
                                1, 21,  2, 2,
                                1, 201, 2, 2]},
                    {},
                    {},
                    {},
                    {},
                ]
            }
        )";
    }
};

BOOST_FIXTURE_TEST_CASE( ParseDetectionPostProcess, DetectionPostProcessFixture )
{
    Setup();

    // Inputs
    using UnquantizedContainer = std::vector<float>;
    UnquantizedContainer boxEncodings =
    {
        0.0f,  0.0f, 0.0f, 0.0f,
        0.0f,  1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,
        0.0f,  1.0f, 0.0f, 0.0f,
        0.0f,  0.0f, 0.0f, 0.0f
    };

    UnquantizedContainer scores =
    {
        0.0f, 0.9f,  0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f,  0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f,  0.4f,
        0.0f, 0.3f,  0.2f
    };

    // Outputs
    UnquantizedContainer detectionBoxes =
    {
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f,  0.0f, 0.0f
    };

    UnquantizedContainer detectionClasses = { 1.0f,  0.0f,  0.0f };
    UnquantizedContainer detectionScores  = { 0.95f, 0.93f, 0.0f };

    UnquantizedContainer numDetections    = { 2.0f };

    // Quantize inputs and outputs
    using QuantizedContainer = std::vector<uint8_t>;
    QuantizedContainer quantBoxEncodings = QuantizedVector<uint8_t>(1.0f, 1, boxEncodings);
    QuantizedContainer quantScores = QuantizedVector<uint8_t>(0.01f, 0, scores);

    std::map<std::string, QuantizedContainer> input =
    {
        { "box_encodings", quantBoxEncodings },
        { "scores", quantScores }
    };

    std::map<std::string, UnquantizedContainer> output =
    {
        { "detection_boxes", detectionBoxes},
        { "detection_classes", detectionClasses},
        { "detection_scores", detectionScores},
        { "num_detections", numDetections}
    };

    RunTest<armnn::DataType::QuantisedAsymm8, armnn::DataType::Float32>(0, input, output);
}

BOOST_FIXTURE_TEST_CASE(DetectionPostProcessGraphStructureTest, DetectionPostProcessFixture)
{
    /*
       Inputs:            box_encodings  scores
                               \          /
                            DetectionPostProcess
                          /        /     \       \
                         /        /       \       \
       Outputs:     detection detection detection num_detections
                    boxes     classes   scores
    */

    ReadStringToBinary();

    armnn::INetworkPtr network = m_Parser->CreateNetworkFromBinary(m_GraphBinary);

    auto optimized = Optimize(*network, { armnn::Compute::CpuRef }, m_Runtime->GetDeviceSpec());

    auto optimizedNetwork = boost::polymorphic_downcast<armnn::OptimizedNetwork*>(optimized.get());
    auto graph = optimizedNetwork->GetGraph();

    // Check the number of layers in the graph
    BOOST_TEST((graph.GetNumInputs() == 2));
    BOOST_TEST((graph.GetNumOutputs() == 4));
    BOOST_TEST((graph.GetNumLayers() == 7));

    // Input layers
    armnn::Layer* boxEncodingLayer = GetFirstLayerWithName(graph, "box_encodings");
    BOOST_TEST((boxEncodingLayer->GetType() == armnn::LayerType::Input));
    BOOST_TEST(CheckNumberOfInputSlot(boxEncodingLayer, 0));
    BOOST_TEST(CheckNumberOfOutputSlot(boxEncodingLayer, 1));

    armnn::Layer* scoresLayer = GetFirstLayerWithName(graph, "scores");
    BOOST_TEST((scoresLayer->GetType() == armnn::LayerType::Input));
    BOOST_TEST(CheckNumberOfInputSlot(scoresLayer, 0));
    BOOST_TEST(CheckNumberOfOutputSlot(scoresLayer, 1));

    // DetectionPostProcess layer
    armnn::Layer* detectionPostProcessLayer = GetFirstLayerWithName(graph, "DetectionPostProcess:0:0");
    BOOST_TEST((detectionPostProcessLayer->GetType() == armnn::LayerType::DetectionPostProcess));
    BOOST_TEST(CheckNumberOfInputSlot(detectionPostProcessLayer, 2));
    BOOST_TEST(CheckNumberOfOutputSlot(detectionPostProcessLayer, 4));

    // Output layers
    armnn::Layer* detectionBoxesLayer = GetFirstLayerWithName(graph, "detection_boxes");
    BOOST_TEST((detectionBoxesLayer->GetType() == armnn::LayerType::Output));
    BOOST_TEST(CheckNumberOfInputSlot(detectionBoxesLayer, 1));
    BOOST_TEST(CheckNumberOfOutputSlot(detectionBoxesLayer, 0));

    armnn::Layer* detectionClassesLayer = GetFirstLayerWithName(graph, "detection_classes");
    BOOST_TEST((detectionClassesLayer->GetType() == armnn::LayerType::Output));
    BOOST_TEST(CheckNumberOfInputSlot(detectionClassesLayer, 1));
    BOOST_TEST(CheckNumberOfOutputSlot(detectionClassesLayer, 0));

    armnn::Layer* detectionScoresLayer = GetFirstLayerWithName(graph, "detection_scores");
    BOOST_TEST((detectionScoresLayer->GetType() == armnn::LayerType::Output));
    BOOST_TEST(CheckNumberOfInputSlot(detectionScoresLayer, 1));
    BOOST_TEST(CheckNumberOfOutputSlot(detectionScoresLayer, 0));

    armnn::Layer* numDetectionsLayer = GetFirstLayerWithName(graph, "num_detections");
    BOOST_TEST((numDetectionsLayer->GetType() == armnn::LayerType::Output));
    BOOST_TEST(CheckNumberOfInputSlot(numDetectionsLayer, 1));
    BOOST_TEST(CheckNumberOfOutputSlot(numDetectionsLayer, 0));

    // Check the connections
    armnn::TensorInfo boxEncodingTensor(armnn::TensorShape({ 1, 6, 4 }), armnn::DataType::QuantisedAsymm8, 1, 1);
    armnn::TensorInfo scoresTensor(armnn::TensorShape({ 1, 6, 3 }), armnn::DataType::QuantisedAsymm8,
                                                      0.00999999978f, 0);

    armnn::TensorInfo detectionBoxesTensor(armnn::TensorShape({ 1, 3, 4 }), armnn::DataType::Float32, 0, 0);
    armnn::TensorInfo detectionClassesTensor(armnn::TensorShape({ 1, 3 }), armnn::DataType::Float32, 0, 0);
    armnn::TensorInfo detectionScoresTensor(armnn::TensorShape({ 1, 3 }), armnn::DataType::Float32, 0, 0);
    armnn::TensorInfo numDetectionsTensor(armnn::TensorShape({ 1} ), armnn::DataType::Float32, 0, 0);

    BOOST_TEST(IsConnected(boxEncodingLayer, detectionPostProcessLayer, 0, 0, boxEncodingTensor));
    BOOST_TEST(IsConnected(scoresLayer, detectionPostProcessLayer, 0, 1, scoresTensor));
    BOOST_TEST(IsConnected(detectionPostProcessLayer, detectionBoxesLayer, 0, 0, detectionBoxesTensor));
    BOOST_TEST(IsConnected(detectionPostProcessLayer, detectionClassesLayer, 1, 0, detectionClassesTensor));
    BOOST_TEST(IsConnected(detectionPostProcessLayer, detectionScoresLayer, 2, 0, detectionScoresTensor));
    BOOST_TEST(IsConnected(detectionPostProcessLayer, numDetectionsLayer, 3, 0, numDetectionsTensor));
}

BOOST_AUTO_TEST_SUITE_END()
