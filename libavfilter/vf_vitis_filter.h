#ifndef AVFILTER_VITIS_FILTER_H
#define AVFILTER_VITIS_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "libavformat/avformat.h"

typedef enum {
    DDMT_SSD,
    DDMT_YOLOV1V2,
    DDMT_YOLOV3,
    DDMT_YOLOV4
} DNNDetectionModelType;

typedef struct DnnContext {
    char *model_filename;
    char *model_inputname;
    char *model_outputnames_string;
    char *backend_options;
    //int async;

    char **model_outputnames;
    uint32_t nb_outputs;
    //const DNNModule *dnn_module;
    //DNNModel *model;
} DnnContext;

typedef struct Yolov3_Ctx {
    char *model_filename;
    char *model_inputname;
    char *model_outputnames_string;
    char *backend_options;
    char **model_outputnames;
    uint32_t nb_outputs;
} Yolov3_Ctx;

typedef struct Yolov8_Ctx {
    char *model_filename;
    char *model_inputname;
    char *model_outputnames_string;
    char *backend_options;
    char **model_outputnames;
    uint32_t nb_outputs;
} Yolov8_Ctx;

typedef struct VitisFilterContext {
    //const AVClass *class;
    DnnContext dnnctx;
    Yolov3_Ctx yolov3ctx;
    Yolov8_Ctx yolov8ctx;
    float confidence;
    //void* model;
    char *labels_filename;
    char **labels;
    int label_count;
    DNNDetectionModelType model_type;
    int cell_w;
    int cell_h;
    int nb_classes;
    //AVFifo *bboxes_fifo;
    int scale_width;
    int scale_height;
    char *anchors_str;
    float *anchors;
    int nb_anchor;
} VitisFilterContext; 

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};

av_cold int vitis_filter_init(AVFilterContext *context);
int vitis_filter_frame(AVFilterLink *inlink, AVFrame *in); //legacy, use activate function instead
int vitis_filter_activate(AVFilterContext *filter_ctx);
av_cold void vitis_filter_uninit(AVFilterContext *context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AVFILTER_VITIS_FILTER_H */



