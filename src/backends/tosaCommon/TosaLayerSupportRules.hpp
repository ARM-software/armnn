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
