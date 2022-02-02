//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/QuantizeHelper.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

#include <vector>

TEST_SUITE("QuantizeHelper")
{
namespace
{

template<typename T>
bool IsFloatIterFunc(T iter)
{
    armnn::IgnoreUnused(iter);
    return armnnUtils::IsFloatingPointIterator<T>::value;
}

} // anonymous namespace

TEST_CASE("IsFloatIterFuncTest")
{
    std::vector<float> fArray;
    CHECK(IsFloatIterFunc(fArray.begin()) == true);
    CHECK(IsFloatIterFunc(fArray.cbegin()) == true);

    std::vector<double> dArray;
    CHECK(IsFloatIterFunc(dArray.begin()) == true);

    std::vector<int> iArray;
    CHECK(IsFloatIterFunc(iArray.begin()) == false);

    float floats[5];
    CHECK(IsFloatIterFunc(&floats[0]) == true);

    int ints[5];
    CHECK(IsFloatIterFunc(&ints[0]) == false);
}

}
