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
    //#include "dnn_filter_common.h"
    #include "internal.h"
    #include "video.h"
    #include "libavutil/time.h"
    #include "libavutil/avstring.h"
    #include "libavutil/detection_bbox.h"
//    #include "libavutil/log.h"
}

#include "vitis/yolov8_onnx_avframe.hpp"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "vitis/color.hpp"

typedef enum {
    DDMT_SSD,
    DDMT_YOLOV1V2,
    DDMT_YOLOV3,
    DDMT_YOLOV4
} DNNDetectionModelType;

/* typedef struct DnnContext {
    //char *model_filename;
    char *model_inputname;
    //char *model_outputnames_string;
    char *backend_options;
    int async;

    char **model_outputnames;
    uint32_t nb_outputs;
    //const DNNModule *dnn_module;
    //DNNModel *model;
} DnnContext;

typedef struct VitisFilterContext {
    const AVClass *class;
    DnnContext dnnctx;
    float confidence;
    //char *labels_filename;
    char **labels;
    int label_count;
    DNNDetectionModelType model_type;
    int cell_w;
    int cell_h;
    int nb_classes;
    AVFifo *bboxes_fifo;
    int scale_width;
    int scale_height;
    char *anchors_str;
    float *anchors;
    int nb_anchor;
} VitisFilterContext; */

//#define OFFSET(x) offsetof(VitisFilterContext, dnnctx.x)
//#define OFFSET2(x) offsetof(VitisFilterContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption vitis_filter_options[] = {
    DNN_COMMON_OPTIONS
    { "confidence",  "threshold of confidence",    OFFSET2(confidence),      AV_OPT_TYPE_FLOAT,     { .dbl = 0.5 },  0, 1, FLAGS},
    { "labels",      "path to labels file",        OFFSET2(labels_filename), AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { "target",      "which one to be classified", OFFSET2(target),          AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(vitis_filter_class);

static void vitis_filter_process_result(cv::Mat& image, const Yolov8OnnxResult& result) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;

    std::cout << "result: " << label << "\t"  << classes[label] << "\t" << std::fixed << std::setprecision(2)
         << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
         << std::setprecision(4) << res.score << "\n";
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
              cv::Scalar(b[label], g[label], r[label]), 3, 1, 0);
    cv::putText(image, classes[label] + " " + std::to_string(res.score),
                    cv::Point(box[0] + 5, box[1] + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(b[label], g[label], r[label]), 2, 4);
                    // cv::Scalar(230, 216, 173), 2, 4);
  }
  return;
}

static av_cold int vitis_filter_init(AVFilterContext *context)
{
    //VitisFilterContext *ctx = context->priv;

    return 0;
}

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};

static int vitis_filter_flush_frame(AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
    //VitisFilterContext *ctx = outlink->src->priv;
    
    return 0;
}

static int vitis_filter_activate(AVFilterContext *filter_ctx)
{
    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    //VitisFilterContext *ctx = filter_ctx->priv;
    AVFrame *in = NULL;
    int64_t pts;
    int ret, status;
    int got_frame = 0;
    //int async_state;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    char* mode_name = ctx->dnnctx.model_name;

    //load models and create filters
    //std::cout << "load model " << argv[1] << endl;
    auto model = Yolov8Onnx::create(std::string(mode_name), 0.3);
    if (!model) {  // supress coverity complain
        //std::cout << "failed to create model\n";
        av_log(NULL, AV_LOG_ERROR, "failed to create model\n");
        return 0;
    }

    std::vector<cv::Mat> images(1);

    do {
        // drain all input frames
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0) 
            //avframe to vitis tensor data
            images[0] = avframeToCvmat(in);

            __TIC__(ONNX_RUN)
            auto results = model->run(images);
            __TOC__(ONNX_RUN)
            
            __TIC__(SHOW)
            vitis_filter_process_result(images[0], results[0]);
            //cv::imshow("yolov8-camera", images[0]);
            __TOC__(SHOW)

            in = cvmatToAvframe(images[0],in);
            ret = ff_filter_frame(outlink, in);
    } while (ret > 0);

    /* // drain all processed frames
    do {
        AVFrame *in_frame = NULL;
        AVFrame *out_frame = NULL;
        async_state = ff_vitis_get_result(&ctx->dnnctx, &in_frame, &out_frame);
        if (async_state == DAST_SUCCESS) {
            ret = ff_filter_frame(outlink, in_frame);
            if (ret < 0)
                return ret;
            got_frame = 1;
        }
    } while (async_state == DAST_SUCCESS);

    // if frame got, schedule to next filter
    if (got_frame)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;
            ret = vitis_filter_flush_frame(outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink); */

    return 0;
}

static av_cold void vitis_filter_uninit(AVFilterContext *context)
{
    //VitisFilterContext *ctx = context->priv;
    
}

static const AVFilterPad vitis_filter_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
};

const AVFilter ff_vf_vitis_filter = {
    .name          = "vitis_filter",
    .description   = NULL_IF_CONFIG_SMALL("Apply Vitis filter to the input."),
    //.priv_size     = sizeof(VitisFilterContext),
    .init          = vitis_filter_init,
    .uninit        = vitis_filter_uninit,
    FILTER_INPUTS(vitis_filter_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .priv_class    = &vitis_filter_class,
    .activate      = vitis_filter_activate,
};
