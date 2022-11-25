//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

// List of Layer Support Rules common to TOSA backends only, for use with CheckSupportRule()

struct TosaOperatorAttributeOfAny : public Rule
{
    template<typename Container>
    explicit TosaOperatorAttributeOfAny(TosaSerializationOperator* op, const Container& c)
    {
        m_Res = std::any_of(c.begin(), c.end(), [&op](Attribute attribute)
        {
            return attribute == op->GetAttributeType();
        });
    }
};

struct TosaTypeAnyOf : public Rule
{
    template<typename Container>
    TosaTypeAnyOf(TosaSerializationTensor* tensor, const Container& c)
    {
        m_Res = std::any_of(c.begin(), c.end(), [&tensor](DType dt)
        {
            return dt == tensor->GetDtype();
        });
    }
};

struct TosaTensorNumDimensionsWithinBounds : public Rule
{
    explicit TosaTensorNumDimensionsWithinBounds(TosaSerializationTensor* tensor)
    {
        m_Res = (tensor->GetShape().size() <= MaxNumOfTensorDimensions) || (!tensor->GetShape().empty());
    }
};

struct TosaAssertSize : public Rule
{
    template<typename Container>
    explicit TosaAssertSize(const Container& c1, const Container& c2)
    {
        m_Res = (c1.size() == c2.size());
    }
};

struct TosaContainerContainsTwoTypes : public Rule
{
    explicit TosaContainerContainsTwoTypes(std::tuple<DType, DType>& check,
                                           const std::vector<std::tuple<DType, DType>>& c)
    {
        for (auto item: c)
        {
            if (std::get<0>(check) == std::get<0>(item) &&
                std::get<1>(check) == std::get<1>(item))
            {
                m_Res = true;
                return;
            }
        }
        m_Res = false;
    }
};

struct TosaContainerContainsThreeTypes : public Rule
{
    explicit TosaContainerContainsThreeTypes(std::tuple<DType, DType, DType>& check,
                                             const std::vector<std::tuple<DType, DType, DType>>& c)
    {
        for (auto item: c)
        {
            if (std::get<0>(check) == std::get<0>(item) &&
                std::get<1>(check) == std::get<1>(item) &&
                std::get<2>(check) == std::get<2>(item))
            {
                m_Res = true;
                return;
            }
        }
        m_Res = false;
    }
};
