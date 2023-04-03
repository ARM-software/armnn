//
// Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"
#include "ParserPrototxtFixture.hpp"
#include "ParserHelper.hpp"
#include <GraphUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/QuantizeHelper.hpp>

TEST_SUITE("TensorflowLiteParser_DetectionPostProcess")
{
struct DetectionPostProcessFixture : ParserFlatbuffersFixture
{
    explicit DetectionPostProcessFixture(const std::string& custom_options)
    {
        /*
            The following values were used for the custom_options:
            use_regular_nms = true
            max_classes_per_detection = 1
            detections_per_class = 1
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
                            "type": "FLOAT32",
                            "buffer": 3,
                            "name": "detection_boxes",
                            "quantization": {}
                        },
                        {
                            "type": "FLOAT32",
                            "buffer": 4,
                            "name": "detection_classes",
                            "quantization": {}
                        },
                        {
                            "type": "FLOAT32",
                            "buffer": 5,
                            "name": "detection_scores",
                            "quantization": {}
                        },
                        {
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
                        "custom_options": [)" + custom_options + R"(],
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

struct ParseDetectionPostProcessCustomOptions : DetectionPostProcessFixture
{
private:
    static armnn::DetectionPostProcessDescriptor GenerateDescriptor()
    {
        static armnn::DetectionPostProcessDescriptor descriptor;
        descriptor.m_UseRegularNms          = true;
        descriptor.m_MaxDetections          = 3u;
        descriptor.m_MaxClassesPerDetection = 1u;
        descriptor.m_DetectionsPerClass     = 1u;
        descriptor.m_NumClasses             = 2u;
        descriptor.m_NmsScoreThreshold      = 0.0f;
        descriptor.m_NmsIouThreshold        = 0.5f;
        descriptor.m_ScaleH                 = 5.0f;
        descriptor.m_ScaleW                 = 5.0f;
        descriptor.m_ScaleX                 = 10.0f;
        descriptor.m_ScaleY                 = 10.0f;

        return descriptor;
    }

public:
    ParseDetectionPostProcessCustomOptions()
        : DetectionPostProcessFixture(
            GenerateDetectionPostProcessJsonString(GenerateDescriptor()))
    {}
};

TEST_CASE_FIXTURE(ParseDetectionPostProcessCustomOptions, "ParseDetectionPostProcess")
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

    QuantizedContainer quantBoxEncodings = armnnUtils::QuantizedVector<uint8_t>(boxEncodings, 1.00f, 1);
    QuantizedContainer quantScores       = armnnUtils::QuantizedVector<uint8_t>(scores,       0.01f, 0);

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

    RunTest<armnn::DataType::QAsymmU8, armnn::DataType::Float32>(0, input, output);
}

TEST_CASE_FIXTURE(ParseDetectionPostProcessCustomOptions, "DetectionPostProcessGraphStructureTest")
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

    armnn::Graph& graph = GetGraphForTesting(optimized.get());

    // Check the number of layers in the graph
    CHECK((graph.GetNumInputs() == 2));
    CHECK((graph.GetNumOutputs() == 4));
    CHECK((graph.GetNumLayers() == 7));

    // Input layers
    armnn::Layer* boxEncodingLayer = GetFirstLayerWithName(graph, "box_encodings");
    CHECK((boxEncodingLayer->GetType() == armnn::LayerType::Input));
    CHECK(CheckNumberOfInputSlot(boxEncodingLayer, 0));
    CHECK(CheckNumberOfOutputSlot(boxEncodingLayer, 1));

    armnn::Layer* scoresLayer = GetFirstLayerWithName(graph, "scores");
    CHECK((scoresLayer->GetType() == armnn::LayerType::Input));
    CHECK(CheckNumberOfInputSlot(scoresLayer, 0));
    CHECK(CheckNumberOfOutputSlot(scoresLayer, 1));

    // DetectionPostProcess layer
    armnn::Layer* detectionPostProcessLayer = GetFirstLayerWithName(graph, "DetectionPostProcess:0:0");
    CHECK((detectionPostProcessLayer->GetType() == armnn::LayerType::DetectionPostProcess));
    CHECK(CheckNumberOfInputSlot(detectionPostProcessLayer, 2));
    CHECK(CheckNumberOfOutputSlot(detectionPostProcessLayer, 4));

    // Output layers
    armnn::Layer* detectionBoxesLayer = GetFirstLayerWithName(graph, "detection_boxes");
    CHECK((detectionBoxesLayer->GetType() == armnn::LayerType::Output));
    CHECK(CheckNumberOfInputSlot(detectionBoxesLayer, 1));
    CHECK(CheckNumberOfOutputSlot(detectionBoxesLayer, 0));

    armnn::Layer* detectionClassesLayer = GetFirstLayerWithName(graph, "detection_classes");
    CHECK((detectionClassesLayer->GetType() == armnn::LayerType::Output));
    CHECK(CheckNumberOfInputSlot(detectionClassesLayer, 1));
    CHECK(CheckNumberOfOutputSlot(detectionClassesLayer, 0));

    armnn::Layer* detectionScoresLayer = GetFirstLayerWithName(graph, "detection_scores");
    CHECK((detectionScoresLayer->GetType() == armnn::LayerType::Output));
    CHECK(CheckNumberOfInputSlot(detectionScoresLayer, 1));
    CHECK(CheckNumberOfOutputSlot(detectionScoresLayer, 0));

    armnn::Layer* numDetectionsLayer = GetFirstLayerWithName(graph, "num_detections");
    CHECK((numDetectionsLayer->GetType() == armnn::LayerType::Output));
    CHECK(CheckNumberOfInputSlot(numDetectionsLayer, 1));
    CHECK(CheckNumberOfOutputSlot(numDetectionsLayer, 0));

    // Check the connections
    armnn::TensorInfo boxEncodingTensor(armnn::TensorShape({ 1, 6, 4 }), armnn::DataType::QAsymmU8, 1, 1);
    armnn::TensorInfo scoresTensor(armnn::TensorShape({ 1, 6, 3 }), armnn::DataType::QAsymmU8,
                                                      0.00999999978f, 0);

    armnn::TensorInfo detectionBoxesTensor(armnn::TensorShape({ 1, 3, 4 }), armnn::DataType::Float32);
    armnn::TensorInfo detectionClassesTensor(armnn::TensorShape({ 1, 3 }), armnn::DataType::Float32);
    armnn::TensorInfo detectionScoresTensor(armnn::TensorShape({ 1, 3 }), armnn::DataType::Float32);
    armnn::TensorInfo numDetectionsTensor(armnn::TensorShape({ 1 } ), armnn::DataType::Float32);

    CHECK(IsConnected(boxEncodingLayer, detectionPostProcessLayer, 0, 0, boxEncodingTensor));
    CHECK(IsConnected(scoresLayer, detectionPostProcessLayer, 0, 1, scoresTensor));
    CHECK(IsConnected(detectionPostProcessLayer, detectionBoxesLayer, 0, 0, detectionBoxesTensor));
    CHECK(IsConnected(detectionPostProcessLayer, detectionClassesLayer, 1, 0, detectionClassesTensor));
    CHECK(IsConnected(detectionPostProcessLayer, detectionScoresLayer, 2, 0, detectionScoresTensor));
    CHECK(IsConnected(detectionPostProcessLayer, numDetectionsLayer, 3, 0, numDetectionsTensor));
}

}
