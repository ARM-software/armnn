//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>

#include <armnn/Optional.hpp>
#include <string>

#include <armnn/utility/IgnoreUnused.hpp>

namespace
{

void PassStringRef(armnn::Optional<std::string&> value)
{
    armnn::IgnoreUnused(value);
}

void PassStringRefWithDefault(armnn::Optional<std::string&> value = armnn::EmptyOptional())
{
    armnn::IgnoreUnused(value);
}

} // namespace <anonymous>

TEST_SUITE("OptionalTests")
{
TEST_CASE("SimpleStringTests")
{
    armnn::Optional<std::string> optionalString;
    CHECK(static_cast<bool>(optionalString) == false);
    CHECK(optionalString.has_value() == false);
    CHECK((optionalString == armnn::Optional<std::string>()));

    optionalString = std::string("Hello World");
    CHECK(static_cast<bool>(optionalString) == true);
    CHECK(optionalString.has_value() == true);
    CHECK(optionalString.value() == "Hello World");
    CHECK((optionalString == armnn::Optional<std::string>("Hello World")));

    armnn::Optional<std::string> otherString;
    otherString = optionalString;
    CHECK(static_cast<bool>(otherString) == true);
    CHECK(otherString.value() == "Hello World");

    optionalString.reset();
    CHECK(static_cast<bool>(optionalString) == false);
    CHECK(optionalString.has_value() == false);

    const std::string stringValue("Hello World");
    armnn::Optional<std::string> optionalString2(stringValue);
    CHECK(static_cast<bool>(optionalString2) == true);
    CHECK(optionalString2.has_value() == true);
    CHECK(optionalString2.value() == "Hello World");

    armnn::Optional<std::string> optionalString3(std::move(optionalString2));
    CHECK(static_cast<bool>(optionalString3) == true);
    CHECK(optionalString3.has_value() == true);
    CHECK(optionalString3.value() == "Hello World");
}

TEST_CASE("StringRefTests")
{
    armnn::Optional<std::string&> optionalStringRef{armnn::EmptyOptional()};
    CHECK(optionalStringRef.has_value() == false);

    PassStringRef(optionalStringRef);
    PassStringRefWithDefault();

    armnn::Optional<std::string&> optionalStringRef2 = optionalStringRef;

    std::string helloWorld("Hello World");

    std::string& helloWorldRef = helloWorld;
    armnn::Optional<std::string&> optionalHelloRef = helloWorldRef;
    CHECK(optionalHelloRef.has_value() == true);
    CHECK(optionalHelloRef.value() == "Hello World");

    armnn::Optional<std::string&> optionalHelloRef2 = helloWorld;
    CHECK(optionalHelloRef2.has_value() == true);
    CHECK(optionalHelloRef2.value() == "Hello World");

    armnn::Optional<std::string&> optionalHelloRef3{helloWorldRef};
    CHECK(optionalHelloRef3.has_value() == true);
    CHECK(optionalHelloRef3.value() == "Hello World");

    armnn::Optional<std::string&> optionalHelloRef4{helloWorld};
    CHECK(optionalHelloRef4.has_value() == true);
    CHECK(optionalHelloRef4.value() == "Hello World");

    // modify through the optional reference
    optionalHelloRef4.value().assign("Long Other String");
    CHECK(helloWorld == "Long Other String");
    CHECK(optionalHelloRef.value() == "Long Other String");
    CHECK(optionalHelloRef2.value() == "Long Other String");
    CHECK(optionalHelloRef3.value() == "Long Other String");
}

TEST_CASE("SimpleIntTests")
{
    const int intValue = 123;

    armnn::Optional<int> optionalInt;
    CHECK(static_cast<bool>(optionalInt) == false);
    CHECK(optionalInt.has_value() == false);
    CHECK((optionalInt == armnn::Optional<int>()));

    optionalInt = intValue;
    CHECK(static_cast<bool>(optionalInt) == true);
    CHECK(optionalInt.has_value() == true);
    CHECK(optionalInt.value() == intValue);
    CHECK((optionalInt == armnn::Optional<int>(intValue)));

    armnn::Optional<int> otherOptionalInt;
    otherOptionalInt = optionalInt;
    CHECK(static_cast<bool>(otherOptionalInt) == true);
    CHECK(otherOptionalInt.value() == intValue);
}

TEST_CASE("ObjectConstructedInPlaceTests")
{
    struct SimpleObject
    {
        public:
            SimpleObject(const std::string& name, int value)
                : m_Name(name)
                , m_Value(value)
            {}

            bool operator ==(const SimpleObject& other)
            {
                return m_Name  == other.m_Name &&
                       m_Value == other.m_Value;
            }

        private:
            std::string m_Name;
            int         m_Value;
    };

    std::string objectName("SimpleObject");
    int objectValue = 1;
    SimpleObject referenceObject(objectName, objectValue);

    // Use MakeOptional
    armnn::Optional<SimpleObject> optionalObject1 = armnn::MakeOptional<SimpleObject>(objectName, objectValue);
    CHECK(static_cast<bool>(optionalObject1) == true);
    CHECK(optionalObject1.has_value() == true);
    CHECK((optionalObject1.value() == referenceObject));

    // Call in-place constructor directly
    armnn::Optional<SimpleObject> optionalObject2(CONSTRUCT_IN_PLACE, objectName, objectValue);
    CHECK(static_cast<bool>(optionalObject1) == true);
    CHECK(optionalObject1.has_value() == true);
    CHECK((optionalObject1.value() == referenceObject));
}

}
