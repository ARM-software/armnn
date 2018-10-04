//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <armnn/Optional.hpp>
#include <string>

BOOST_AUTO_TEST_SUITE(OptionalTests)

BOOST_AUTO_TEST_CASE(SimpleStringTests)
{
    armnn::Optional<std::string> optionalString;
    BOOST_TEST(optionalString == false);
    BOOST_TEST(optionalString.has_value() == false);

    optionalString = std::string("Hello World");
    BOOST_TEST(optionalString == true);
    BOOST_TEST(optionalString.has_value() == true);
    BOOST_TEST(optionalString.value() == "Hello World");

    armnn::Optional<std::string> otherString;
    otherString = optionalString;
    BOOST_TEST(otherString == true);
    BOOST_TEST(otherString.value() == "Hello World");

    optionalString.reset();
    BOOST_TEST(optionalString == false);
    BOOST_TEST(optionalString.has_value() == false);

    const std::string stringValue("Hello World");
    armnn::Optional<std::string> optionalString2(stringValue);
    BOOST_TEST(optionalString2 == true);
    BOOST_TEST(optionalString2.has_value() == true);
    BOOST_TEST(optionalString2.value() == "Hello World");

    armnn::Optional<std::string> optionalString3(std::move(optionalString2));
    BOOST_TEST(optionalString3 == true);
    BOOST_TEST(optionalString3.has_value() == true);
    BOOST_TEST(optionalString3.value() == "Hello World");
}

BOOST_AUTO_TEST_CASE(SimpleIntTests)
{
    const int intValue = 123;

    armnn::Optional<int> optionalInt;
    BOOST_TEST(optionalInt == false);
    BOOST_TEST(optionalInt.has_value() == false);

    optionalInt = intValue;
    BOOST_TEST(optionalInt == true);
    BOOST_TEST(optionalInt.has_value() == true);
    BOOST_TEST(optionalInt.value() == intValue);

    armnn::Optional<int> otherOptionalInt;
    otherOptionalInt = optionalInt;
    BOOST_TEST(otherOptionalInt == true);
    BOOST_TEST(otherOptionalInt.value() == intValue);
}

BOOST_AUTO_TEST_SUITE_END()
