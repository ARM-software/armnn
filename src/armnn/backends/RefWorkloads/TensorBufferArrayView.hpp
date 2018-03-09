//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <armnn/Tensor.hpp>

#include <boost/assert.hpp>

namespace armnn
{

// Utility class providing access to raw tensor memory based on indices along each dimension
template <typename DataType>
class TensorBufferArrayView
{
public:
    TensorBufferArrayView(const TensorShape& shape, DataType* data)
        : m_Shape(shape)
        , m_Data(data)
    {
    }

    DataType& Get(unsigned int b, unsigned int c, unsigned int h, unsigned int w) const
    {
        BOOST_ASSERT( b < m_Shape[0] || (m_Shape[0] == 0 && b == 0) );
        BOOST_ASSERT( c < m_Shape[1] || (m_Shape[1] == 0 && c == 0) );
        BOOST_ASSERT( h < m_Shape[2] || (m_Shape[2] == 0 && h == 0) );
        BOOST_ASSERT( w < m_Shape[3] || (m_Shape[3] == 0 && w == 0) );

        return m_Data[b * m_Shape[1] * m_Shape[2] * m_Shape[3]
                    + c * m_Shape[2] * m_Shape[3]
                    + h * m_Shape[3]
                    + w];
    }

private:
    const TensorShape m_Shape;
    DataType* m_Data;
};

} //namespace armnn
