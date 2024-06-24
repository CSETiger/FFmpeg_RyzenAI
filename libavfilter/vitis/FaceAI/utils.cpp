#include "utils.h"

#include <d3d11.h>
#include <d3dcompiler.h>

using namespace std;
using namespace cv;

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

class DirectComputeWrapper
{
public:
    DirectComputeWrapper();
    ~DirectComputeWrapper();
    bool Initialize();
    void WarpAffine(const Mat& src, Mat& dst, const Mat& M, const Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar());

private:
    ID3D11Device* device;
    ID3D11DeviceContext* context;
    ID3D11ComputeShader* computeShader;
    ID3D11Buffer* constantBuffer;
    ID3D11ShaderResourceView* inputImageSRV;
    ID3D11UnorderedAccessView* outputImageUAV;
};

DirectComputeWrapper::DirectComputeWrapper()
    : device(nullptr), context(nullptr), computeShader(nullptr),
    constantBuffer(nullptr), inputImageSRV(nullptr), outputImageUAV(nullptr)
{
}

DirectComputeWrapper::~DirectComputeWrapper()
{
    if (constantBuffer) constantBuffer->Release();
    if (computeShader) computeShader->Release();
    if (context) context->Release();
    if (device) device->Release();
}

bool DirectComputeWrapper::Initialize()
{
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &device,
        nullptr,
        &context);

    if (FAILED(hr))
    {
        cerr << "Failed to create D3D11 device." << endl;
        return false;
    }

    ID3DBlob* csBlob = nullptr;
    hr = D3DCompileFromFile(L"WarpAffine.hlsl", nullptr, nullptr, "main", "cs_5_0", 0, 0, &csBlob, nullptr);
    if (FAILED(hr))
    {
        if (csBlob){
            OutputDebugStringA(reinterpret_cast<const char*>(csBlob->GetBufferPointer()));
            csBlob->Release();
        }
        cerr << "Failed to compile compute shader, err=0x" << hr << endl;
        return false;
    }

    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);
    csBlob->Release();
    if (FAILED(hr))
    {
        cerr << "Failed to create compute shader." << endl;
        return false;
    }

    // 创建常量缓冲区
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.ByteWidth = sizeof(float) * 16; // 3x3 matrix + width + height + flags + borderMode + borderValue
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    hr = device->CreateBuffer(&cbDesc, nullptr, &constantBuffer);
    if (FAILED(hr))
    {
        cerr << "Failed to create constant buffer." << endl;
        return false;
    }

    return true;
}

void DirectComputeWrapper::WarpAffine(const Mat& inputImage, Mat& outputImage, const Mat& affineMatrix, Size dsize, int flags, int borderMode, const Scalar& borderValue)
{
    // 转换图像为 GPU 可用的格式
    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = inputImage.cols;
    texDesc.Height = inputImage.rows;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = inputImage.data;
    initData.SysMemPitch = static_cast<UINT>(inputImage.step);

    ID3D11Texture2D* inputTexture = nullptr;
    device->CreateTexture2D(&texDesc, &initData, &inputTexture);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;

    device->CreateShaderResourceView(inputTexture, &srvDesc, &inputImageSRV);

    texDesc.Width = dsize.width;
    texDesc.Height = dsize.height;
    texDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ID3D11Texture2D* outputTexture = nullptr;
    device->CreateTexture2D(&texDesc, nullptr, &outputTexture);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = texDesc.Format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;

    device->CreateUnorderedAccessView(outputTexture, &uavDesc, &outputImageUAV);

    // 设置常量缓冲区
    float cbData[16] = {
        affineMatrix.at<float>(0, 0), affineMatrix.at<float>(0, 1), affineMatrix.at<float>(0, 2),
        affineMatrix.at<float>(1, 0), affineMatrix.at<float>(1, 1), affineMatrix.at<float>(1, 2),
        0, 0, 1,
        static_cast<float>(dsize.width),
        static_cast<float>(dsize.height),
        static_cast<float>(flags),
        static_cast<float>(borderMode),
        borderValue[0], borderValue[1], borderValue[2]
    };

    context->UpdateSubresource(constantBuffer, 0, nullptr, cbData, 0, 0);

    context->CSSetShader(computeShader, nullptr, 0);
    context->CSSetConstantBuffers(0, 1, &constantBuffer);
    context->CSSetShaderResources(0, 1, &inputImageSRV);
    context->CSSetUnorderedAccessViews(0, 1, &outputImageUAV, nullptr);

    context->Dispatch((dsize.width + 15) / 16, (dsize.height + 15) / 16, 1);

    // 读取结果
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    context->Map(outputTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
    memcpy(outputImage.data, mappedResource.pData, outputImage.step * outputImage.rows);
    context->Unmap(outputTexture, 0);

    inputTexture->Release();
    outputTexture->Release();
}

float GetIoU(const Bbox box1, const Bbox box2)
{
    float x1 = max(box1.xmin, box2.xmin);
    float y1 = max(box1.ymin, box2.ymin);
    float x2 = min(box1.xmax, box2.xmax);
    float y2 = min(box1.ymax, box2.ymax);
    float w = max(0.f, x2 - x1);
    float h = max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

vector<int> nms(vector<Bbox> boxes, vector<float> confidences, const float nms_thresh)
{
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
         { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}

Mat warp_face_by_face_landmark_5(const Mat temp_vision_frame, Mat &crop_img, const vector<Point2f> face_landmark_5, const vector<Point2f> normed_template, const Size crop_size)
{
    vector<uchar> inliers(face_landmark_5.size(), 0);
    Mat affine_matrix = cv::estimateAffinePartial2D(face_landmark_5, normed_template, cv::noArray(), cv::RANSAC, 100.0);
    warpAffine(temp_vision_frame, crop_img, affine_matrix, crop_size, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return affine_matrix;
}

Mat create_static_box_mask(const int *crop_size, const float face_mask_blur, const int *face_mask_padding)
{
    const float blur_amount = int(crop_size[0] * 0.5 * face_mask_blur);
    const int blur_area = max(int(blur_amount / 2), 1);
    Mat box_mask = Mat::ones(crop_size[0], crop_size[1], CV_32FC1);

    int sub = max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100));
    // Mat roi = box_mask(cv::Rect(0,0,sub,crop_size[1]));
    box_mask(cv::Rect(0, 0, crop_size[1], sub)).setTo(0);

    sub = crop_size[0] - max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100));
    box_mask(cv::Rect(0, sub, crop_size[1], crop_size[0] - sub)).setTo(0);

    sub = max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100));
    box_mask(cv::Rect(0, 0, sub, crop_size[0])).setTo(0);

    sub = crop_size[1] - max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100));
    box_mask(cv::Rect(sub, 0, crop_size[1] - sub, crop_size[0])).setTo(0);

    if (blur_amount > 0)
    {
        GaussianBlur(box_mask, box_mask, Size(0, 0), blur_amount * 0.25);
    }
    return box_mask;
}

Mat paste_back(Mat temp_vision_frame, Mat crop_vision_frame, Mat crop_mask, Mat affine_matrix)
{
    Mat inverse_matrix;
    DirectComputeWrapper dcw;
    if (!dcw.Initialize())
    {
        cerr << "Failed to initialize DirectCompute." << endl;
        return inverse_matrix;
    }

    cv::invertAffineTransform(affine_matrix, inverse_matrix);
    Mat inverse_mask;
    Size temp_size(temp_vision_frame.cols, temp_vision_frame.rows);
    //warpAffine(crop_mask, inverse_mask, inverse_matrix, temp_size);
    dcw.WarpAffine(crop_mask, inverse_mask, inverse_matrix, temp_size);
    inverse_mask.setTo(0, inverse_mask < 0);
    inverse_mask.setTo(1, inverse_mask > 1);
    Mat inverse_vision_frame;
    //warpAffine(crop_vision_frame, inverse_vision_frame, inverse_matrix, temp_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    dcw.WarpAffine(crop_vision_frame, inverse_vision_frame, inverse_matrix, temp_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    vector<Mat> inverse_vision_frame_bgrs(3);
    split(inverse_vision_frame, inverse_vision_frame_bgrs);
    vector<Mat> temp_vision_frame_bgrs(3);
    split(temp_vision_frame, temp_vision_frame_bgrs);
    for (int c = 0; c < 3; c++)
    {
        inverse_vision_frame_bgrs[c].convertTo(inverse_vision_frame_bgrs[c], CV_32FC1);   ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
        temp_vision_frame_bgrs[c].convertTo(temp_vision_frame_bgrs[c], CV_32FC1);         ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
    }
    vector<Mat> channel_mats(3);
    
    channel_mats[0] = inverse_mask.mul(inverse_vision_frame_bgrs[0]) + temp_vision_frame_bgrs[0].mul(1 - inverse_mask);
    channel_mats[1] = inverse_mask.mul(inverse_vision_frame_bgrs[1]) + temp_vision_frame_bgrs[1].mul(1 - inverse_mask);
    channel_mats[2] = inverse_mask.mul(inverse_vision_frame_bgrs[2]) + temp_vision_frame_bgrs[2].mul(1 - inverse_mask);
    
    cv::Mat paste_vision_frame;
    merge(channel_mats, paste_vision_frame);
    paste_vision_frame.convertTo(paste_vision_frame, CV_8UC3);
    return paste_vision_frame;
}

Mat blend_frame(Mat temp_vision_frame, Mat paste_vision_frame, const int FACE_ENHANCER_BLEND)
{
    const float face_enhancer_blend = 1 - ((float)FACE_ENHANCER_BLEND / 100.f);
    Mat dstimg;
    cv::addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0, dstimg);
    return dstimg;
}