#include "libavutil/file_open.h"
#include "libavutil/opt.h"
#include "filters.h"
#include "avfilter.h"
//#include "dnn_filter_common.h"
#include "internal.h"
#include "video.h"
#include "libavutil/time.h"
#include "libavutil/avstring.h"
#include "libavutil/detection_bbox.h"
#include "vf_vitis_filter.h"


#define OFFSET(x) offsetof(VitisFilterContext, dnnctx.x)
#define OFFSET2(x) offsetof(VitisFilterContext, x)
#define OFFSET3(x) offsetof(VitisFilterContext, faceswapctx.x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

#define RYZENAI_OPTIONS \
    { "model",              "path to model file",         OFFSET(model_filename),   AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },\
    { "ep",    "options of execution provider",            OFFSET(ep_name),  AV_OPT_TYPE_STRING,    { .str = "VitisAI" }, 0, 0, FLAGS },\
    //{ "input",              "input name of the model",    OFFSET(model_inputname),  AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },\
    //{ "output",             "output name of the model",   OFFSET(model_outputnames_string), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },\
    { "backend_configs",    "backend configs",            OFFSET(backend_options),  AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },\
    { "options", "backend configs (deprecated, use backend_configs)", OFFSET(backend_options),  AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS | AV_OPT_FLAG_DEPRECATED},\
    //{ "async",              "use DNN async inference (ignored, use backend_configs='async=1')",    OFFSET(async),            AV_OPT_TYPE_BOOL,      { .i64 = 1},     0, 1, FLAGS},

static const AVOption vitis_filter_options[] = {
    RYZENAI_OPTIONS
    { "confidence",  "threshold of confidence",    OFFSET2(confidence),      AV_OPT_TYPE_FLOAT,     { .dbl = 0.3 },  0, 1, FLAGS},
    { "facemodelpath",   "path to models of Face AI",         OFFSET3(modelpath),   AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { "sourceimg",       "path to source image file",         OFFSET3(source_image),   AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    //{ "labels",      "path to labels file",        OFFSET2(labels_filename), AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    //{ "target",      "which one to be classified", OFFSET2(target),          AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(vitis_filter);

static const AVFilterPad vitis_filter_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        //.config_props = config_input,
    },
};

static const AVFilterPad vitis_filter_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
};

const AVFilter ff_vf_vitis_filter = {
    .name          = "vitis_filter",
    .description   = NULL_IF_CONFIG_SMALL("Apply Ryzen AI Effects to the video."),
    .priv_size     = sizeof(VitisFilterContext),
    .init          = vitis_filter_init,
    .uninit        = vitis_filter_uninit,
    FILTER_INPUTS(vitis_filter_inputs),
    FILTER_OUTPUTS(vitis_filter_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .priv_class    = &vitis_filter_class,
    .activate      = vitis_filter_activate,
};