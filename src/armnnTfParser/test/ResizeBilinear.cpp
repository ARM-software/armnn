//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ResizeBilinearFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ResizeBilinearFixture()
    {
        m_Prototext = R"(
node {
  name: "graphInput"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 1
          }
        }
        tensor_content:
"\000\000\000\000\000\000\200?\000\000\000@\000\000@@\000\000\200@\000\000\240@\000\000\300@\000\000\340@\000\000\000A"
      }
    }
  }
}
node {
  name: "resizeBilinearLayer/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000"
      }
    }
  }
}
node {
  name: "resizeBilinearLayer"
  op: "ResizeBilinear"
  input: "graphInput"
  input: "resizeBilinearLayer/size"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "align_corners"
    value {
      b: false
    }
  }
}
        )";

        SetupSingleInputSingleOutput({ 1, 3, 3, 1 }, "graphInput", "resizeBilinearLayer");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseResizeBilinear, ResizeBilinearFixture)
{
    RunTest<4>(// Input data.
               { 0.0f, 1.0f, 2.0f,
                 3.0f, 4.0f, 5.0f,
                 6.0f, 7.0f, 8.0f },
               // Expected output data.
               { 0.0f, 0.6f, 1.2f, 1.8f, 2.0f,
                 1.8f, 2.4f, 3.0f, 3.6f, 3.8f,
                 3.6f, 4.2f, 4.8f, 5.4f, 5.6f,
                 5.4f, 6.0f, 6.6f, 7.2f, 7.4f,
                 6.0f, 6.6f, 7.2f, 7.8f, 8.0f });

}

BOOST_AUTO_TEST_SUITE_END()
