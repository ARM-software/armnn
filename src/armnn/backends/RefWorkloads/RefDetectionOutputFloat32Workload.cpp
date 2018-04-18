
#include "RefDetectionOutputFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <iostream>

namespace armnn
{
    using uint = unsigned int;

    template <typename Dtype>
    class Box{
    public:
        Dtype x;
        Dtype y;
        Dtype w;
        Dtype h;

        inline void chop()
        {
            Dtype xmin = std::max(Dtype(0), (x - w/2.0f));
            x = xmin;
            Dtype xmax = std::min(Dtype(1), (x + w/2.0f));
            w = xmax;
            Dtype ymin = std::max(Dtype(0), (y - h/2.0f));
            y = ymin;
            Dtype ymax = std::min(Dtype(1), (y + h/2.0f));
            h = ymax;
        }
        inline Dtype area()
        {
            return (w - x)*(h - y);
        }
    };

    template <typename Dtype>
    class PredictionResult : public Box<Dtype>{
    public:
        int   classType;
        Dtype confidence;
        Dtype objScore;
    };

    template <typename Dtype>
    inline Dtype sigmoid(Dtype x)
    {
        return static_cast<Dtype>( 1. / (1. + exp(-x)) );
    }

    template <typename Dtype>
    inline void get_region_box(Dtype* x, PredictionResult<Dtype>& predict, std::vector<Dtype> biases,
                        uint n, uint index, uint i, uint j, uint w, uint h)
    {
        Dtype rw = 1 / static_cast<Dtype>(w);
        Dtype rh = 1 / static_cast<Dtype>(h);
        predict.x = ( static_cast<Dtype>(i) + sigmoid(x[index + 0]) ) * rw;
        predict.y = ( static_cast<Dtype>(j) + sigmoid(x[index + 1]) ) * rh;
        predict.w = static_cast<Dtype>( exp(x[index + 2]) * biases[2*n] * rw );
        predict.h = static_cast<Dtype>( exp(x[index + 3]) * biases[2*n+1] * rh );
    }

    // do multiply obj softmax with all classes and get max scode id.
    template <typename Dtype>
    Dtype class_index_and_score(Dtype* input, uint classes, PredictionResult<Dtype>& predict)
    {
        Dtype* tail = input + classes;
        Dtype large = *std::max_element(input ,tail);

        std::for_each(input ,tail ,[large](Dtype &d){
            Dtype e = static_cast<Dtype>( exp(d - large) );
            d = e;
        });
        Dtype sum = std::accumulate(input ,tail ,0.f);
        sum = 1.f/sum;
        std::for_each(input ,tail ,[sum](Dtype &d){  d *= sum; });

        auto pos = std::max_element(input ,tail);
        large = *pos;
        predict.classType = static_cast<int>( std::distance(input,pos) );

        return large;
    }

    template<typename DType>
    DType CalculateOverlap( Box<DType> a,  Box<DType> b)
    {
        a.chop();
        b.chop();

        DType w = std::max(DType(0), std::min(a.w, b.w) - std::max(a.x, b.x));
        DType h = std::max(DType(0), std::min(a.h, b.h) - std::max(a.y, b.y));
        DType i = w * h;
        DType u = a.area() + b.area() - i;
        return u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
    }

    // object pair with iou above nms_threshold treat as one object
    void applyNms(std::vector< PredictionResult<float> > &predicts,float nms_threshold)
    {
        uint valid_count = static_cast<uint>(predicts.size());

        for (uint i = 0; i < valid_count; ++i)
        {
            if (predicts[i].classType < 0)
                continue;  // skip  already eliminated
            for (uint j = i + 1; j < valid_count; ++j)
            {
                if (predicts[j].classType < 0)
                    continue;

                if ( predicts[j].classType == predicts[i].classType )  // classType equals
                {
                    float iou = CalculateOverlap( predicts[i], predicts[j]);
                    if (iou >= nms_threshold)
                    {
                        predicts[j].classType = -1;
                    }
                }
            }
        }
    }

    void RefDetectionOutputFloat32Workload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDetectionOutputFloat32Workload_Execute");

        float*        outputData = GetOutputTensorDataFloat(0, m_Data);
        const float* inputData  = GetInputTensorDataFloat(0, m_Data);

        TensorInfo  inInfo = GetTensorInfo(m_Data.m_Inputs[0]);
        TensorShape inShape = inInfo.GetShape();

        const unsigned int classes = m_Data.m_Parameters.m_Classes;
        const unsigned int side    = m_Data.m_Parameters.m_Side;
        const unsigned int num_box = m_Data.m_Parameters.m_NumBox;
        const unsigned int coords = m_Data.m_Parameters.m_Coords;

        std::vector<float> biases = m_Data.m_Parameters.m_Biases;
        const float confidence_threshold = m_Data.m_Parameters.m_ConfidenceThreshold;
        const float nms_threshold = m_Data.m_Parameters.m_NmsThreshold;

        std::cout << " confidence_threshold: " << confidence_threshold << std::endl;
        std::cout << " nms_threshold: " << nms_threshold << std::endl;

        const uint num = inShape[0];
        const uint height = inShape[2];
        const uint width = inShape[3];
        const uint channels = inShape[1];

        std::vector<float> swap(num*height*width*channels);

        // permute
        auto data_at = [inputData, height, width, channels](uint b,uint c,uint h,uint w)->float
        {   return  inputData[ b*height*width*channels + c*height*width + h*width + w ];   };

        // dims[num,channel,h,w] --> dims[num,h,w,channel]
        uint index = 0;
        for (uint b = 0 ; b < num ; ++b )
            for (uint h = 0 ; h < height; ++h )
                for (uint w = 0 ; w < width ; ++w )
                    for (uint c = 0 ; c < channels ; ++c )
                        swap[index++] = data_at(b, c, h, w);

        std::vector< PredictionResult<float> > predicts;
        PredictionResult<float> predict;
        predicts.clear();
        float objScore;
        float classScore;
        uint group_size = classes + 1 + coords;
        for (uint b = 0; b < num; ++b)
        {
            for (uint j = 0; j < side; ++j)
                for (uint i = 0; i < side; ++i)
                    for (uint n = 0; n < num_box; ++n)
                    {
                        uint index = b * channels * height * width +
                                    (j * side + i) * num_box * group_size + n * group_size;

                        get_region_box<float>(swap.data(), predict, biases, n, index, i, j, side, side);

                        objScore = sigmoid(swap[index + 4]);
                        classScore = class_index_and_score(swap.data() + index + 5, classes, predict);

                        predict.confidence = objScore * classScore;
                        predict.objScore = objScore;

                        if (predict.confidence >= confidence_threshold)
                            predicts.push_back(predict);
                    }

            if (predicts.size() > 0)
                applyNms(predicts, nms_threshold);

            uint num_kept = 0;
            for (uint i = 0; i < predicts.size(); i++) {
                if (predicts[i].classType != -1) {
                    outputData[i * 7] = static_cast<float>(b);                         //Image_Id
                    outputData[i * 7 + 1] = static_cast<float>(predicts[i].classType); //label
                    outputData[i * 7 + 2] = predicts[i].confidence;                      //confidence
                    outputData[i * 7 + 3] = predicts[i].x;
                    outputData[i * 7 + 4] = predicts[i].y;
                    outputData[i * 7 + 5] = predicts[i].w;
                    outputData[i * 7 + 6] = predicts[i].h;
                    num_kept++;
                }
            }
            std::vector<uint> outShape(2, 1);
            outShape.push_back(num_kept);
            outShape.push_back(7);
        }
    }

} //namespace armnn
