#include "FaceAI/yolov8face.h"
#include "FaceAI/face68landmarks.h"
#include "FaceAI/facerecognizer.h"
#include "FaceAI/faceswap.h"
#include "FaceAI/faceenhancer.h"


using namespace cv;
using namespace std;

class faceswap_onnx {
    public:
        static std::unique_ptr<faceswap_onnx> create(const std::string& modelpath) {
            return std::unique_ptr<faceswap_onnx>(
                new faceswap_onnx(modelpath));
        }
        int faceswap_detect_src(string& source_image);
        int faceswap_process(cv::Mat& target_img);

        Yolov8Face* detect_face_net;
        Face68Landmarks* detect_68landmarks_net;
        FaceEmbdding* face_embedding_net;
        SwapFace* swap_face_net;
        FaceEnhance* enhance_face_net;

    protected:
        explicit faceswap_onnx(const std::string& modelpath);

    private:
        vector<Bbox> boxes;
        vector<Point2f> face_landmark_5of68;
        vector<Point2f> face68landmarks;
        vector<float> source_face_embedding;
        vector<Point2f> target_landmark_5;
};

faceswap_onnx::faceswap_onnx(const std::string& modelpath)
{
    detect_face_net = new Yolov8Face(modelpath + "/yoloface_8n.onnx");
    detect_68landmarks_net = new Face68Landmarks(modelpath + "/2dfan4.onnx");
    face_embedding_net = new FaceEmbdding(modelpath + "/arcface_w600k_r50.onnx");
    swap_face_net = new SwapFace(modelpath + "/inswapper_128.onnx");
    enhance_face_net = new FaceEnhance(modelpath + "/gfpgan_1.4.onnx");

}

int faceswap_onnx::faceswap_detect_src(string& source_image)
{
    cv::Mat source_img = imread(source_image);

	detect_face_net->detect(source_img, boxes);
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况

	face68landmarks = detect_68landmarks_net->detect(source_img, boxes[position], face_landmark_5of68);
	source_face_embedding = face_embedding_net->detect(source_img, face_landmark_5of68);

    return 0;
}

int faceswap_onnx::faceswap_process(cv::Mat& target_img)
{    
	detect_face_net->detect(target_img, boxes);
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	
	detect_68landmarks_net->detect(target_img, boxes[position], target_landmark_5);

	Mat swapimg = swap_face_net->process(target_img, source_face_embedding, target_landmark_5);
	Mat resultimg = enhance_face_net->process(swapimg, target_landmark_5);
	
    return 0;

}