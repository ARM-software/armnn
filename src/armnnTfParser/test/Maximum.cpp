//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct MaximumFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MaximumFixture(const armnn::TensorShape& inputShape0, const armnn::TensorShape& inputShape1)
    {
        m_Prototext = R"(
node {
  name: "input0"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "input1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "output"
  op: "Maximum"
  input: "input0"
  input: "input1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
        )";

        Setup({ { "input0", inputShape0 },
                { "input1", inputShape1 } },
              { "output" });
    }
};

struct MaximumFixture4D4D : public MaximumFixture
{
    MaximumFixture4D4D() : MaximumFixture({ 1, 2, 2, 3 }, { 1, 2, 2, 3 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMaximum4D4D, MaximumFixture4D4D)
{
    RunTest<4>({ { "input0", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } },
                 { "input1", { 5.0f, 1.0f, 3.0f,
                               4.0f, 5.5f, 1.0f,
                               2.0f, 17.0f, 18.0f,
                               19.0f, 1.0f, 3.0f } } },
               { { "output", { 5.0f,  1.0f, 3.0f,
                               4.0f,  5.5f, 5.0f,
                               6.0f,  17.0f, 18.0f,
                               19.0f, 10.0f, 11.0f } } });
}

struct MaximumBroadcastFixture4D4D : public MaximumFixture
{
    MaximumBroadcastFixture4D4D() : MaximumFixture({ 1, 1, 2, 1 }, { 1, 2, 1, 3 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMaximumBroadcast4D4D, MaximumBroadcastFixture4D4D)
{
    RunTest<4>({ { "input0", { 2.0f, 4.0f } },
                 { "input1", { 1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f } } },
               { { "output", { 2.0f, 2.0f, 3.0f,
                               4.0f, 4.0f, 4.0f,
                               4.0f, 5.0f, 6.0f,
                               4.0f, 5.0f, 6.0f } } });
}

struct MaximumBroadcastFixture4D1D : public MaximumFixture
{
    MaximumBroadcastFixture4D1D() : MaximumFixture({ 1, 2, 2, 3 }, { 1 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMaximumBroadcast4D1D, MaximumBroadcastFixture4D1D)
{
    RunTest<4>({ { "input0", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } },
                 { "input1", { 5.0f } } },
               { { "output", { 5.0f, 5.0f, 5.0f,
                               5.0f, 5.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } } });
}

struct MaximumBroadcastFixture1D4D : public MaximumFixture
{
    MaximumBroadcastFixture1D4D() : MaximumFixture({ 1 }, { 1, 2, 2, 3 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMaximumBroadcast1D4D, MaximumBroadcastFixture1D4D)
{
    RunTest<4>({ { "input0", { 3.0f } },
                 { "input1", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } } },
               { { "output", { 3.0f, 3.0f, 3.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
