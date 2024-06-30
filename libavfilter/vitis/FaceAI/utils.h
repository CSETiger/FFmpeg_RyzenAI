# ifndef UTILS
# define UTILS
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/directx.hpp>

#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/experimental_onnxruntime_cxx_api.h>
#include <core/session/dml_provider_factory.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#  include <codecvt>
#  include <locale>

using namespace cv;
using namespace std;
using namespace Ort;

// #define GLOG_USE_GLOG_EXPORT
// #define GOOGLE_GLOG_DLL_DECL
// #define GLOG_NO_ABBREVIATED_SEVERITIES
// //#include <glog/logging.h>

// #include "../onnx_task.hpp"

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} Bbox;

typedef struct
{
    UINT32 width;
    UINT32 height;
    float flags;
    float borderMode;
    float borderValue[4];
    float affinematrix[3][2]; // 2 used
    UINT32 srcwidth;
    UINT32 srcheight; //2 used
    float pad[4];
} ConstantBuffer;

class DirectComputeWrapper
{
public:
    DirectComputeWrapper();
    ~DirectComputeWrapper();
    bool Initialize();
    bool DC_init;
    void WarpAffine(const Mat& src, Mat& dst, const Mat& M, const Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar());

private:
    ID3D11Device* device;
    ID3D11DeviceContext* context;
    ID3D11ComputeShader* computeShader;
    ID3D11Buffer* constantBuffer;
    ID3D11ShaderResourceView* inputImageSRV;
    ID3D11UnorderedAccessView* outputImageUAV;
};

float GetIoU(const Bbox box1, const Bbox box2);
std::vector<int> nms(std::vector<Bbox> boxes, std::vector<float> confidences, const float nms_thresh);
cv::Mat warp_face_by_face_landmark_5(const cv::Mat temp_vision_frame, cv::Mat &crop_img, const std::vector<cv::Point2f> face_landmark_5, const std::vector<cv::Point2f> normed_template, const cv::Size crop_size);
cv::Mat create_static_box_mask(const int *crop_size, const float face_mask_blur, const int *face_mask_padding);
cv::Mat paste_back(cv::Mat temp_vision_frame, cv::Mat crop_vision_frame, cv::Mat crop_mask, cv::Mat affine_matrix);
cv::Mat blend_frame(cv::Mat temp_vision_frame, cv::Mat paste_vision_frame, const int FACE_ENHANCER_BLEND=80);
#endif