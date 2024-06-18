#include "yolov8face.h"

Yolov8Face::Yolov8Face(string model_path, const float conf_thres, const float iou_thresh)
{
    constexpr GraphOptimizationLevel GRAPH_OPTIMIZATION_LEVEL = GraphOptimizationLevel::ORT_ENABLE_ALL;
    using convert_t = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_t, wchar_t> strconverter2;
    /// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释
    printf("Yolov8Face entering----->\n");
    // sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    // std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    // ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    // //ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法
    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
      const OrtDmlApi* ortDmlApi;
      //THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
      if (ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)) != nullptr){
        //av_log(NULL, AV_LOG_ERROR, "OnnxTask: GetEP failed---\n");
        printf("Yolov8Face GetEP failed----->\n");
      }

      //av_log(NULL, AV_LOG_INFO, "OnnxTask: GetEP -------\n");
      printf("Yolov8Face GetEP done----->\n");

      //Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "Onnx DirectML"); 
      sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      sessionOptions.DisableMemPattern();
      sessionOptions.SetGraphOptimizationLevel(GRAPH_OPTIMIZATION_LEVEL);
      //av_log(NULL, AV_LOG_INFO, "OnnxTask: AppendEP_DML -------\n");
      printf("Yolov8Face AppendEP_DML----->\n");
      ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, /*device index*/ 0);
        auto model_name_basic = strconverter2.from_bytes(model_path);
      session_.reset(
        new Ort::Experimental::Session(env, model_name_basic, sessionOptions));
        
    size_t numInputNodes = session_->GetInputCount();
    size_t numOutputNodes = session_->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        //input_names.push_back(ort_session->GetInputName(i, allocator));      ///低版本onnxruntime的接口函数
        ////AllocatedStringPtr input_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
        ////input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        //output_names.push_back(ort_session->GetOutputName(i, allocator));  ///低版本onnxruntime的接口函数
        ////AllocatedStringPtr output_name_Ptr= ort_session->GetInputNameAllocated(i, allocator);
        ////output_names.push_back(output_name_Ptr.get()); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
    this->conf_threshold = conf_thres;
    this->iou_threshold = iou_thresh;
}

void Yolov8Face::preprocess(Mat srcimg)
{
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    Mat temp_image = srcimg.clone();
    if (height > this->input_height || width > this->input_width)
    {
        const float scale = min((float)this->input_height / height, (float)this->input_width / width);
        Size new_size = Size(int(width * scale), int(height * scale));
        resize(srcimg, temp_image, new_size);
    }
    this->ratio_height = (float)height / temp_image.rows;
    this->ratio_width = (float)width / temp_image.cols;
    Mat input_img;
    copyMakeBorder(temp_image, input_img, 0, this->input_height - temp_image.rows, 0, this->input_width - temp_image.cols, BORDER_CONSTANT, 0);

    vector<cv::Mat> bgrChannels(3);
    split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int image_area = this->input_height * this->input_width;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

////只返回检测框,因为在下游的模块里,置信度和5个关键点这两个信息在后续的模块里没有用到
void Yolov8Face::detect(Mat srcimg, std::vector<Bbox> &boxes)
{
    this->preprocess(srcimg);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    //std::vector<float> input_tensor_values;
    std::vector<Ort::Value> input_tensor_;
    std::vector<Ort::Value> ort_outputs;
    //Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());
    if (input_tensor_.size()) {
    input_tensor_[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_image.data(), input_image.size(),
        input_img_shape);
    } else {
        input_tensor_.push_back(Ort::Experimental::Value::CreateTensor<float>(
            input_image.data(), input_image.size(),
            input_img_shape));
    }

    //Ort::RunOptions runOptions;
    //vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());
    ort_outputs = this->session_->Run(session_->GetInputNames(), input_tensor_, session_->GetOutputNames());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 20, 8400),不考虑第0维batchsize，每一列的长度20,前4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下的15个元素是5个关键点坐标x,y和置信度
    const int num_box = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    vector<Bbox> bounding_box_raw;
    vector<float> score_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = pdata[4 * num_box + i];
        if (score > this->conf_threshold)
        {
            float xmin = (pdata[i] - 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[num_box + i] - 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[num_box + i] + 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一下
            bounding_box_raw.emplace_back(Bbox{xmin, ymin, xmax, ymax});
            score_raw.emplace_back(score);
            /// 剩下的5个关键点坐标的计算,暂时不写,因为在下游的模块里没有用到5个关键点坐标信息
        }
    }
    vector<int> keep_inds = nms(bounding_box_raw, score_raw, this->iou_threshold);
    const int keep_num = keep_inds.size();
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }
}