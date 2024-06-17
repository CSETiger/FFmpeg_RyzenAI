/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * implementing an AI filter using RyzenAI.
 */
extern "C"{
    #include "libavutil/file_open.h"
    #include "libavutil/opt.h"
    #include "filters.h"
    #include "avfilter.h"
    #include "internal.h"
    #include "video.h"
    #include "libavutil/time.h"
    #include "libavutil/avstring.h"
    #include "libavutil/detection_bbox.h"
    #include "vf_vitis_filter.h"
}

#pragma comment(lib, "glog.lib") 
#pragma comment(lib, "opencv_world490.lib") 
#pragma comment(lib, "onnxruntime.lib") 
#pragma comment(lib, "onnxruntime_providers_shared.lib") 

#include "vitis/yolov8_onnx_avframe.hpp"
#include "vitis/faceswap_onnx.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "vitis/color.hpp"

//namespace std {
using namespace cv;

std::unique_ptr<Yolov8Onnx> Yolov8OnnxModel;
std::unique_ptr<faceswap_onnx> FaceSwapOnnx;

extern "C"{
void vitis_filter_process_result(cv::Mat& image, const Yolov8OnnxResult& result) {
    av_log(NULL, AV_LOG_INFO, "vitis filter: vitis_filter_process_result ------->\n");
    for (auto& res : result.bboxes) {
        int label = res.label;
        auto& box = res.box;

        /* std::cout << "result: " << label << "\t"  << classes[label] << "\t" << std::fixed << std::setprecision(2)
            << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
            << std::setprecision(4) << res.score << "\n"; */
        cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
                cv::Scalar(b[label], g[label], r[label]), 3, 1, 0);
        cv::putText(image, classes[label] + " " + std::to_string(res.score),
                        cv::Point(box[0] + 5, box[1] + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(b[label], g[label], r[label]), 2, 4);
                        // cv::Scalar(230, 216, 173), 2, 4);
    }
    return;
}


av_cold int vitis_filter_init(AVFilterContext *context)
{
    av_log(NULL, AV_LOG_INFO, "vitis filter: vitis_filter_init entering---->\n");

    VitisFilterContext *ctx = (VitisFilterContext *)context->priv;
    //model.reset(nullptr);
    char* model_name = ctx->dnnctx.model_filename;
    char* ep_name = ctx->dnnctx.ep_name;
    //load models and create filters
    //std::cout << "load model " << argv[1] << endl;
    av_log(NULL, AV_LOG_INFO, "vitis filter::model_name:%s ep_name:%s\n",model_name, ep_name);
    auto model = Yolov8Onnx::create(std::string(model_name), ctx->confidence, std::string(ep_name));
    //ctx->model = model.get();
    if (!model) {  // supress coverity complain
        av_log(NULL, AV_LOG_ERROR, "vitis filter: failed to create model\n");
        return -1;
    }
    av_log(NULL, AV_LOG_INFO, "vitis filter: model created\n");
    Yolov8OnnxModel = std::move(model);

    //faceswap
    char* facemodelpath = ctx->faceswapctx.modelpath;
    char* sourceimg = ctx->faceswapctx.source_image;
    FaceSwapOnnx = faceswap_onnx::create(facemodelpath);
    //faceswap_load_models(facemodelpath);
    FaceSwapOnnx->faceswap_detect_src(std::string(sourceimg));

    return 0;
}

/*use filter activate instead of filter frame*/
int vitis_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    av_log(NULL, AV_LOG_INFO, "vitis filter: vitis_filter_frame entering---->\n");
    
    return 0;
}

int vitis_filter_activate(AVFilterContext *filter_ctx)
{
    av_log(NULL, AV_LOG_INFO, "vitis filter: vitis_filter_activate entering---->\n");

    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    VitisFilterContext *ctx = (VitisFilterContext *)filter_ctx->priv;
    AVFrame *in_frame = NULL;
    AVFrame *out_frame = NULL;
    int64_t pts;
    int ret = 0; 
    int status = 0;
    std::vector<cv::Mat> images(1);
    //char* mode_name = ctx->dnnctx.model_filename;
    //int got_frame = 0;
    //int async_state;

    //FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    //load models and create filters
    //std::cout << "load model " << argv[1] << endl;
    //auto model = Yolov8Onnx::create(std::string(mode_name), ctx->confidence);
    if (!Yolov8OnnxModel) {  // supress coverity complain
        //std::cout << "failed to create model\n";
        av_log(NULL, AV_LOG_ERROR, "vitis filter: failed to create model\n");
        return 0;
    }

    // drain all input frames
    ret = ff_inlink_consume_frame(inlink, &in_frame);
    av_log(NULL, AV_LOG_INFO, "vitis filter: ff_inlink_consume_frame return %d\n",ret);
    if (ret < 0){
        av_log(NULL, AV_LOG_INFO, "vitis filter: ff_inlink_consume_frame failed\n");
        return ret;
    }
    if (in_frame) {
        //avframe to vitis tensor data
        av_log(NULL, AV_LOG_INFO, "vitis filter: avframeToCvmat\n");
        images[0] = avframeToCvmat(in_frame);

        __TIC__(ONNX_RUN)
        av_log(NULL, AV_LOG_INFO, "vitis filter: Yolov8OnnxModel->run(images) begin\n");
        auto results = Yolov8OnnxModel->run(images);
        av_log(NULL, AV_LOG_INFO, "vitis filter: Yolov8OnnxModel->run(images) end\n");
        __TOC__(ONNX_RUN)
        
        __TIC__(SHOW)
        vitis_filter_process_result(images[0], results[0]);
        
        FaceSwapOnnx->faceswap_process(images[0]);//process face swapping

        __TOC__(SHOW)
        av_log(NULL, AV_LOG_INFO, "vitis filter: cvmatToAvframe begin\n");
        out_frame = cvmatToAvframe(&images[0],in_frame);
        av_log(NULL, AV_LOG_INFO, "vitis filter: cvmatToAvframe end\n");
        ret = ff_filter_frame(outlink, out_frame);
        av_log(NULL, AV_LOG_INFO, "vitis filter: ff_filter_frame return %d\n",ret);
        if (ret < 0){
            av_frame_free(&out_frame);
            return ret;
        }
    }
    else{
        av_log(NULL, AV_LOG_INFO, "vitis filter: ff_inlink_request_frame\n");
        ff_inlink_request_frame(inlink);
    }

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            //int64_t out_pts = pts;
            //ret = vitis_filter_flush_frame(outlink, pts, &out_pts);
            av_log(NULL, AV_LOG_INFO, "vitis filter: ff_inlink_acknowledge_status AVERROR_EOF\n");
            ff_outlink_set_status(outlink, status, pts);
            return 0;
        }
    }
    
    //FF_FILTER_FORWARD_WANTED(outlink, inlink); 

    return 0;
}

av_cold void vitis_filter_uninit(AVFilterContext *context)
{
    av_log(NULL, AV_LOG_INFO, "vitis filter: vitis_filter_uninit entering---->\n");
    VitisFilterContext *ctx = (VitisFilterContext *)context->priv;
    if (Yolov8OnnxModel){
        auto model = Yolov8OnnxModel.release();
    }
}

} //extern "C"
//}
//} //namespace ai
//} //namespace vitis