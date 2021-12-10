//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Reshape_Dynamic")
{
struct ReshapeDynamicFixture1 : public ParserFlatbuffersFixture
{
    explicit ReshapeDynamicFixture1()
    {
        m_JsonString = R"(
{
  "version": 3,
  "operator_codes": [
    {
      "deprecated_builtin_code": 77,
      "version": 1,
      "builtin_code": "ADD"
    },
    {
      "deprecated_builtin_code": 22,
      "version": 1,
      "builtin_code": "ADD"
    }
  ],
  "subgraphs": [
    {
      "tensors": [
        {
          "shape": [
            2,
            5
          ],
          "type": "FLOAT32",
          "buffer": 1,
          "name": "input_33",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false,
          "shape_signature": [
            -1,
            10
          ]
        },
        {
          "shape": [
            2
          ],
          "type": "INT32",
          "buffer": 2,
          "name": "functional_15/tf_op_layer_Shape_9/Shape_9",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false
        },
        {
          "shape": [
            2,
            5
          ],
          "type": "FLOAT32",
          "buffer": 3,
          "name": "Identity",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false,
          "shape_signature": [
            -1,
            5
          ]
        }
      ],
      "inputs": [
        0
      ],
      "outputs": [
        2
      ],
      "operators": [
        {
          "opcode_index": 0,
          "inputs": [
            0
          ],
          "outputs": [
            1
          ],
          "builtin_options_type": "ShapeOptions",
          "builtin_options": {
            "out_type": "INT32"
          },
          "custom_options_format": "FLEXBUFFERS"
        },
        {
          "opcode_index": 1,
          "inputs": [
            0,
            1
          ],
          "outputs": [
            2
          ],
          "builtin_options_type": "NONE",
          "custom_options_format": "FLEXBUFFERS"
        }
      ],
      "name": "main"
    }
  ],
  "description": "MLIR Converted.",
  "buffers": [
    {
    },
    {
    },
    {
    },
    {
    },
    {
      "data": [
        49,
        46,
        49,
        48,
        46,
        48,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    }
  ],
  "metadata": [
    {
      "name": "min_runtime_version",
      "buffer": 4
    }
  ]
}
)";

    }
};

// ParseReshape test case: reshapeShapes[0] == 2
TEST_CASE_FIXTURE(ReshapeDynamicFixture1, "ParseReshapeDynamic1")
{
    SetupSingleInputSingleOutput("input_33", "Identity");
    RunTest<2, armnn::DataType::Float32>(0,
                                         { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                                         { 1, 2, 3, 4, 5,
                                           6, 7, 8, 9, 10 });

    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "Identity").second.GetShape()
        == armnn::TensorShape({2,5})));
}

struct ReshapeDynamicFixture2 : public ParserFlatbuffersFixture
{
    explicit ReshapeDynamicFixture2()
    {
        m_JsonString = R"(
{
  "version": 3,
  "operator_codes": [
    {
      "deprecated_builtin_code": 77,
      "version": 1,
      "builtin_code": "ADD"
    },
    {
      "deprecated_builtin_code": 22,
      "version": 1,
      "builtin_code": "ADD"
    }
  ],
  "subgraphs": [
    {
      "tensors": [
        {
          "shape": [
            -1,
            10
          ],
          "type": "FLOAT32",
          "buffer": 1,
          "name": "input_33",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false,
          "shape_signature": [
            2,
            5
          ]
        },
        {
          "shape": [
            1
          ],
          "type": "INT32",
          "buffer": 2,
          "name": "functional_15/tf_op_layer_Shape_9/Shape_9",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false
        },
        {
          "shape": [
            10
          ],
          "type": "FLOAT32",
          "buffer": 3,
          "name": "Identity",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false,
          "shape_signature": [
            -1,
            10
          ]
        }
      ],
      "inputs": [
        0
      ],
      "outputs": [
        2
      ],
      "operators": [
        {
          "opcode_index": 0,
          "inputs": [
            0
          ],
          "outputs": [
            1
          ],
          "builtin_options_type": "ShapeOptions",
          "builtin_options": {
            "out_type": "INT32"
          },
          "custom_options_format": "FLEXBUFFERS"
        },
        {
          "opcode_index": 1,
          "inputs": [
            0,
            1
          ],
          "outputs": [
            2
          ],
          "builtin_options_type": "NONE",
          "custom_options_format": "FLEXBUFFERS"
        }
      ],
      "name": "main"
    }
  ],
  "description": "MLIR Converted.",
  "buffers": [
    {
    },
    {
    },
    {
    },
    {
    },
    {
      "data": [
        49,
        46,
        49,
        48,
        46,
        48,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    }
  ],
  "metadata": [
    {
      "name": "min_runtime_version",
      "buffer": 4
    }
  ]
}
)";

    }
};

// ParseReshape test case: reshapeShapes[0] == 1
TEST_CASE_FIXTURE(ReshapeDynamicFixture2, "ParseReshapeDynamic2")
{
    SetupSingleInputSingleOutput("input_33", "Identity");
    RunTest<1, armnn::DataType::Float32>(0,
                                         { 1, 2, 3, 4, 5,
                                           6, 7, 8, 9, 10 },
                                         { 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "Identity").second.GetShape()
        == armnn::TensorShape({10})));
}

struct ReshapeDynamicFixture3 : public ParserFlatbuffersFixture
{
    explicit ReshapeDynamicFixture3()
    {
        m_JsonString = R"(
{
  "version": 3,
  "operator_codes": [
    {
      "deprecated_builtin_code": 77,
      "version": 1,
      "builtin_code": "ADD"
    },
    {
      "deprecated_builtin_code": 22,
      "version": 1,
      "builtin_code": "ADD"
    }
  ],
  "subgraphs": [
    {
      "tensors": [
        {
          "shape": [
            1,
            9
          ],
          "type": "FLOAT32",
          "buffer": 1,
          "name": "input_33",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false,
          "shape_signature": [
            -1,
            9
          ]
        },
        {
          "shape": [
            3
          ],
          "type": "INT32",
          "buffer": 2,
          "name": "functional_15/tf_op_layer_Shape_9/Shape_9",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false
        },
        {
          "shape": [
            1,
            9
          ],
          "type": "FLOAT32",
          "buffer": 3,
          "name": "Identity",
          "quantization": {
            "details_type": "NONE",
            "quantized_dimension": 0
          },
          "is_variable": false,
          "shape_signature": [
            -1,
            9
          ]
        }
      ],
      "inputs": [
        0
      ],
      "outputs": [
        2
      ],
      "operators": [
        {
          "opcode_index": 0,
          "inputs": [
            0
          ],
          "outputs": [
            1
          ],
          "builtin_options_type": "ShapeOptions",
          "builtin_options": {
            "out_type": "INT32"
          },
          "custom_options_format": "FLEXBUFFERS"
        },
        {
          "opcode_index": 1,
          "inputs": [
            0,
            1
          ],
          "outputs": [
            2
          ],
          "builtin_options_type": "NONE",
          "custom_options_format": "FLEXBUFFERS"
        }
      ],
      "name": "main"
    }
  ],
  "description": "MLIR Converted.",
  "buffers": [
    {
    },
    {
    },
    {
    },
    {
    },
    {
      "data": [
        49,
        46,
        49,
        48,
        46,
        48,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    }
  ],
  "metadata": [
    {
      "name": "min_runtime_version",
      "buffer": 4
    }
  ]
}
)";

    }
};

// ParseReshape test case: reshapeShapes[0] > 2
TEST_CASE_FIXTURE(ReshapeDynamicFixture3, "ParseReshapeDynamic3")
{
    CHECK_THROWS_AS(SetupSingleInputSingleOutput("input_33", "Identity"), armnn::ParseException);
}

}
