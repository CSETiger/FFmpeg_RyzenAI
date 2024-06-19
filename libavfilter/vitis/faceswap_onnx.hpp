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
        int faceswap_loadmodels();
        void faceswap_unloadmodels();
        int faceswap_detect_src(string& source_image);
        int faceswap_process(cv::Mat& target_img);

    protected:
        explicit faceswap_onnx(const std::string& modelpath);
        faceswap_onnx(const faceswap_onnx&) = delete;

        Yolov8Face* detect_face_net;
        Face68Landmarks* detect_68landmarks_net;
        FaceEmbdding* face_embedding_net;
        SwapFace* swap_face_net;
        FaceEnhance* enhance_face_net; 

    private:
        string mode_path;
        vector<Bbox> boxes;
        vector<Point2f> face_landmark_5of68;
        vector<Point2f> face68landmarks;
        vector<float> source_face_embedding;
        vector<Point2f> target_landmark_5;
};

int faceswap_onnx::faceswap_loadmodels()
{
    printf("faceswap_onnx::faceswap_loadmodels loading models from %s\n", mode_path);

    detect_face_net = new Yolov8Face(mode_path + "/yoloface_8n.onnx");
    if (!detect_face_net){
        printf("faceswap_onnx::faceswap_onnx loading yolov8face from %s failed!\n", mode_path);
        return -1;
    }

    detect_68landmarks_net = new Face68Landmarks(mode_path + "/2dfan4.onnx");
    if (!detect_68landmarks_net){
        printf("faceswap_onnx::faceswap_onnx loading Face68Landmarks from %s failed!\n", mode_path);
        return -2;
    } 

    face_embedding_net = new FaceEmbdding(mode_path + "/arcface_w600k_r50.onnx");
    if (!face_embedding_net){
         printf("faceswap_onnx::faceswap_onnx loading FaceEmbdding from %s failed!\n", mode_path);
         return -3;
    }
       
    swap_face_net = new SwapFace(mode_path + "/inswapper_128.onnx");
    if (!swap_face_net){
        printf("faceswap_onnx::faceswap_onnx loading swap_face_net from %s failed!\n", mode_path);
        return -4;
    }

    enhance_face_net = new FaceEnhance(mode_path + "/gfpgan_1.4.onnx");
    if (!enhance_face_net){
        printf("faceswap_onnx::faceswap_onnx loading enhance_face_net from %s failed!\n", mode_path);
        return -5;
    }

    return 0;
}

void faceswap_onnx::faceswap_unloadmodels()
{
    printf("faceswap_onnx::faceswap_unloadmodels unloading models\n");
    if (detect_face_net){
        delete detect_face_net;
        detect_face_net = nullptr;
    }
    if (detect_68landmarks_net){
        delete detect_68landmarks_net;
        detect_68landmarks_net = nullptr;
    }
    if (face_embedding_net){
        delete face_embedding_net;
        face_embedding_net = nullptr;
    }
    if (swap_face_net){
        delete swap_face_net;
        swap_face_net = nullptr;
    }
    if (enhance_face_net){
        delete enhance_face_net;
        enhance_face_net = nullptr;
    }

}

faceswap_onnx::faceswap_onnx(const std::string& modelpath)
{
    printf("faceswap_onnx::faceswap_onnx setpath %s\n", modelpath);
    //mode_path = "C:/Users/tianwyan/Desktop/DMLtest";
    mode_path = modelpath;
    printf("faceswap_onnx::faceswap_onnx getpath %s\n", mode_path);


    // detect_face_net = new Yolov8Face("weights/yoloface_8n.onnx");
    // if (!detect_face_net)
    //     printf("faceswap_onnx::faceswap_onnx loading yolov8face failed!\n");
    // detect_68landmarks_net = new Face68Landmarks("weights/2dfan4.onnx");
    // if (!detect_68landmarks_net)
    //     printf("faceswap_onnx::faceswap_onnx loading Face68Landmarks failed!\n");
    // face_embedding_net = new FaceEmbdding("weights/arcface_w600k_r50.onnx");
    // if (!face_embedding_net)
    //     printf("faceswap_onnx::faceswap_onnx loading FaceEmbdding failed!\n");
    // swap_face_net = new SwapFace("weights/inswapper_128.onnx");
    // if (!swap_face_net)
    //     printf("faceswap_onnx::faceswap_onnx loading swap_face_net failed!\n");
    // enhance_face_net = new FaceEnhance("weights/gfpgan_1.4.onnx");
    // if (!enhance_face_net)
    //     printf("faceswap_onnx::faceswap_onnx loading enhance_face_net failed!\n");

}

int faceswap_onnx::faceswap_detect_src(string& source_image)
{
    cv::Mat source_img = imread(source_image);

	detect_face_net->detect(source_img, boxes);
    if (!boxes.size()) {
        printf("faceswap_onnx::faceswap_detect_src no faces found in source image!\n");
        return -1;
    }
        
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况

	face68landmarks = detect_68landmarks_net->detect(source_img, boxes[position], face_landmark_5of68);
	source_face_embedding = face_embedding_net->detect(source_img, face_landmark_5of68);

    return 0;
}

int faceswap_onnx::faceswap_process(cv::Mat& target_img)
{    
    if (!source_face_embedding.size()) {
        //printf("faceswap_onnx::faceswap_detect_src skipping!\n");
        return -1;
    }

	detect_face_net->detect(target_img, boxes);
    if (!boxes.size()) {
        //printf("faceswap_onnx::faceswap_process no faces found in target image!\n");
        return -1;
    }

	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	
	detect_68landmarks_net->detect(target_img, boxes[position], target_landmark_5);

	Mat swapimg = swap_face_net->process(target_img, source_face_embedding, target_landmark_5);
	//Mat resultimg = enhance_face_net->process(swapimg, target_landmark_5);
	target_img = enhance_face_net->process(swapimg, target_landmark_5);

    return 0;

}