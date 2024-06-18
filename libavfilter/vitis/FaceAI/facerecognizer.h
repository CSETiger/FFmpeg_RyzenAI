# ifndef FACERECOGNIZER
# define FACERECOGNIZER

#include"utils.h"


class FaceEmbdding
{
public:
	FaceEmbdding(std::string modelpath);
	std::vector<float> detect(cv::Mat srcimg, const std::vector<cv::Point2f> face_landmark_5);
private:
	void preprocess(cv::Mat img, const std::vector<cv::Point2f> face_landmark_5);
	std::vector<float> input_image;
	int input_height;
	int input_width;
    std::vector<cv::Point2f> normed_template;

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Feature Extract");
	Ort::Session *ort_session = nullptr;
	std::unique_ptr<Ort::Experimental::Session> session_;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
#endif