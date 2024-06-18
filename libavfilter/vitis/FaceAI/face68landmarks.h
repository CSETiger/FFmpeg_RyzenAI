# ifndef DETECT_FACE68LANDMARKS
# define DETECT_FACE68LANDMARKS

#include"utils.h"


class Face68Landmarks
{
public:
	Face68Landmarks(std::string modelpath);
	std::vector<cv::Point2f> detect(cv::Mat srcimg, const Bbox bounding_box, std::vector<cv::Point2f> &face_landmark_5of68);
private:
	void preprocess(cv::Mat img, const Bbox bounding_box);
	std::vector<float> input_image;
	int input_height;
	int input_width;
    cv::Mat inv_affine_matrix;

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "68FaceLandMarks Detect");
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