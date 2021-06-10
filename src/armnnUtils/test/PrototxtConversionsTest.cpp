//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <PrototxtConversions.hpp>
#include "armnn/Tensor.hpp"

#include <doctest/doctest.h>

TEST_SUITE("PrototxtConversions")
{
TEST_CASE("ConvertInt32ToOctalStringTest")
{
    using armnnUtils::ConvertInt32ToOctalString;

    std::string octalString = ConvertInt32ToOctalString(1);
    CHECK(octalString.compare("\\\\001\\\\000\\\\000\\\\000"));

    octalString = ConvertInt32ToOctalString(256);
    CHECK(octalString.compare("\\\\000\\\\100\\\\000\\\\000"));

    octalString = ConvertInt32ToOctalString(65536);
    CHECK(octalString.compare("\\\\000\\\\000\\\\100\\\\000"));

    octalString = ConvertInt32ToOctalString(16777216);
    CHECK(octalString.compare("\\\\000\\\\000\\\\000\\\\100"));

    octalString = ConvertInt32ToOctalString(-1);
    CHECK(octalString.compare("\\\\377\\\\377\\\\377\\\\377"));

    octalString = ConvertInt32ToOctalString(-256);
    CHECK(octalString.compare("\\\\000\\\\377\\\\377\\\\377"));

    octalString = ConvertInt32ToOctalString(-65536);
    CHECK(octalString.compare("\\\\000\\\\000\\\\377\\\\377"));

    octalString = ConvertInt32ToOctalString(-16777216);
    CHECK(octalString.compare("\\\\000\\\\000\\\\000\\\\377"));
}

TEST_CASE("ConvertTensorShapeToStringTest")
{
    using armnnUtils::ConvertTensorShapeToString;
    using armnn::TensorShape;

    auto createAndConvert = [](std::initializer_list<unsigned int> dims) -> std::string
    {
        auto shape = TensorShape{dims};
        return ConvertTensorShapeToString(shape);
    };

    auto output_string = createAndConvert({5});
    CHECK(output_string.compare(
        "dim {\n"
        "size: 5\n"
        "}"));

    output_string = createAndConvert({4, 5});
    CHECK(output_string.compare(
        "dim {\n"
            "size: 4\n"
        "}\n"
        "dim {\n"
            "size: 5\n"
        "}"
        ));

    output_string = createAndConvert({3, 4, 5});
    CHECK(output_string.compare(
        "dim {\n"
            "size: 3\n"
        "}\n"
        "dim {\n"
            "size: 4\n"
        "}\n"
        "dim {\n"
            "size: 5\n"
        "}"
        ));

    output_string = createAndConvert({2, 3, 4, 5});
    CHECK(output_string.compare(
        "dim {\n"
            "size: 2\n"
        "}\n"
        "dim {\n"
            "size: 3\n"
        "}\n"
        "dim {\n"
            "size: 4\n"
        "}\n"
        "dim {\n"
            "size: 5\n"
        "}"
        ));

    output_string = createAndConvert({1, 2, 3, 4, 5});
    CHECK(output_string.compare(
        "dim {\n"
            "size: 1\n"
        "}\n"
        "dim {\n"
            "size: 2\n"
        "}\n"
        "dim {\n"
            "size: 3\n"
        "}\n"
        "dim {\n"
            "size: 4\n"
        "}\n"
        "dim {\n"
            "size: 5\n"
        "}"
        ));

    output_string = createAndConvert({0xffffffff, 0xffffffff});
    CHECK(output_string.compare(
        "dim {\n"
            "size: 4294967295\n"
        "}\n"
        "dim {\n"
            "size: 4294967295\n"
        "}"
        ));

    output_string = createAndConvert({1, 0});
    CHECK(output_string.compare(
        "dim {\n"
            "size: 1\n"
        "}\n"
        "dim {\n"
            "size: 0\n"
        "}"
        ));
}

}
