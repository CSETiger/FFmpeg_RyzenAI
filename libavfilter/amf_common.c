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

#include "amf_common.h"

#include "libavutil/avassert.h"
#include "avfilter.h"
#include "internal.h"
#include "formats.h"
#include "libavutil/imgutils.h"

#include "libavutil/hwcontext_amf.h"
#include "AMF/components/ColorSpace.h"
#include "scale_eval.h"

#if CONFIG_DXVA2
#include <d3d9.h>
#endif

#if CONFIG_D3D11VA
#include <d3d11.h>
#endif

int amf_scale_init(AVFilterContext *avctx)
{
    AMFScaleContext     *ctx = avctx->priv;

    if (!strcmp(ctx->format_str, "same")) {
        ctx->format = AV_PIX_FMT_NONE;
    } else {
        ctx->format = av_get_pix_fmt(ctx->format_str);
        if (ctx->format == AV_PIX_FMT_NONE) {
            av_log(avctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", ctx->format_str);
            return AVERROR(EINVAL);
        }
    }

    return 0;
}

void amf_scale_uninit(AVFilterContext *avctx)
{
    AMFScaleContext *ctx = avctx->priv;

    if (ctx->scaler) {
        ctx->scaler->pVtbl->Terminate(ctx->scaler);
        ctx->scaler->pVtbl->Release(ctx->scaler);
        ctx->scaler = NULL;
    }

    av_buffer_unref(&ctx->amf_device_ref);
    av_buffer_unref(&ctx->hwdevice_ref);
    av_buffer_unref(&ctx->hwframes_in_ref);
    av_buffer_unref(&ctx->hwframes_out_ref);
}

int amf_scale_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext             *avctx = inlink->dst;
    AMFScaleContext             *ctx = avctx->priv;
    AVFilterLink                *outlink = avctx->outputs[0];
    AMF_RESULT  res;
    AMFSurface *surface_in;
    AMFSurface *surface_out;
    AMFData *data_out = NULL;
    enum AVColorSpace out_colorspace;
    enum AVColorRange out_color_range;

    AVFrame *out = NULL;
    int ret = 0;

    if (!ctx->scaler)
        return AVERROR(EINVAL);

    ret = amf_avframe_to_amfsurface(avctx, in, &surface_in);
    if (ret < 0)
        goto fail;

    res = ctx->scaler->pVtbl->SubmitInput(ctx->scaler, (AMFData*)surface_in);
    AMF_GOTO_FAIL_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "SubmitInput() failed with error %d\n", res);
    res = ctx->scaler->pVtbl->QueryOutput(ctx->scaler, &data_out);
    AMF_GOTO_FAIL_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "QueryOutput() failed with error %d\n", res);

    if (data_out) {
        AMFGuid guid = IID_AMFSurface();
        data_out->pVtbl->QueryInterface(data_out, &guid, (void**)&surface_out); // query for buffer interface
        data_out->pVtbl->Release(data_out);
    }

    out = amf_amfsurface_to_avframe(avctx, surface_out);

    ret = av_frame_copy_props(out, in);
    av_frame_unref(in);

    out_colorspace = AVCOL_SPC_UNSPECIFIED;

    if (ctx->color_profile != AMF_VIDEO_CONVERTER_COLOR_PROFILE_UNKNOWN) {
        switch(ctx->color_profile) {
        case AMF_VIDEO_CONVERTER_COLOR_PROFILE_601:
            out_colorspace = AVCOL_SPC_SMPTE170M;
        break;
        case AMF_VIDEO_CONVERTER_COLOR_PROFILE_709:
            out_colorspace = AVCOL_SPC_BT709;
        break;
        case AMF_VIDEO_CONVERTER_COLOR_PROFILE_2020:
            out_colorspace = AVCOL_SPC_BT2020_NCL;
        break;
        case AMF_VIDEO_CONVERTER_COLOR_PROFILE_JPEG:
            out_colorspace = AVCOL_SPC_RGB;
        break;
        default:
            out_colorspace = AVCOL_SPC_UNSPECIFIED;
        break;
        }
        out->colorspace = out_colorspace;
    }

    out_color_range = AVCOL_RANGE_UNSPECIFIED;
    if (ctx->color_range == AMF_COLOR_RANGE_FULL)
        out_color_range = AVCOL_RANGE_JPEG;
    else if (ctx->color_range == AMF_COLOR_RANGE_STUDIO)
        out_color_range = AVCOL_RANGE_MPEG;

    if (ctx->color_range != AMF_COLOR_RANGE_UNDEFINED)
        out->color_range = out_color_range;

    if (ctx->primaries != AMF_COLOR_PRIMARIES_UNDEFINED)
        out->color_primaries = ctx->primaries;

    if (ctx->trc != AMF_COLOR_TRANSFER_CHARACTERISTIC_UNDEFINED)
        out->color_trc = ctx->trc;


    if (ret < 0)
        goto fail;

    out->format = outlink->format;
    out->width  = outlink->w;
    out->height = outlink->h;

    out->hw_frames_ctx = av_buffer_ref(ctx->hwframes_out_ref);
    if (!out->hw_frames_ctx) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    //surface_in->pVtbl->Release(surface_in);
    //surface_out->pVtbl->Release(surface_out);

    if (inlink->sample_aspect_ratio.num) {
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h * inlink->w, outlink->w * inlink->h}, inlink->sample_aspect_ratio);
    } else
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}



int amf_setup_input_output_formats(AVFilterContext *avctx,
                                    const enum AVPixelFormat *input_pix_fmts,
                                    const enum AVPixelFormat *output_pix_fmts)
{
    int err;
    int i;
    AVFilterFormats *input_formats;

    //in case if hw_device_ctx is set to DXVA2 we change order of pixel formats to set DXVA2 be choosen by default
    //The order is ignored if hw_frames_ctx is not NULL on the config_output stage
    if (avctx->hw_device_ctx) {
        AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)avctx->hw_device_ctx->data;

        switch (device_ctx->type) {
    #if CONFIG_D3D11VA
        case AV_HWDEVICE_TYPE_D3D11VA:
            {
                static const enum AVPixelFormat output_pix_fmts_d3d11[] = {
                    AV_PIX_FMT_D3D11,
                    AV_PIX_FMT_NONE,
                };
                output_pix_fmts = output_pix_fmts_d3d11;
            }
            break;
    #endif
    #if CONFIG_DXVA2
        case AV_HWDEVICE_TYPE_DXVA2:
            {
                static const enum AVPixelFormat output_pix_fmts_dxva2[] = {
                    AV_PIX_FMT_DXVA2_VLD,
                    AV_PIX_FMT_NONE,
                };
                output_pix_fmts = output_pix_fmts_dxva2;
            }
            break;
    #endif
        default:
            {
                av_log(avctx, AV_LOG_ERROR, "Unsupported device : %s\n", av_hwdevice_get_type_name(device_ctx->type));
                return AVERROR(EINVAL);
            }
            break;
        }
    }

    input_formats = ff_make_format_list(output_pix_fmts);
    if (!input_formats) {
        return AVERROR(ENOMEM);
    }

    for (i = 0; input_pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
        err = ff_add_format(&input_formats, input_pix_fmts[i]);
        if (err < 0)
            return err;
    }

    if ((err = ff_formats_ref(input_formats, &avctx->inputs[0]->outcfg.formats)) < 0 ||
        (err = ff_formats_ref(ff_make_format_list(output_pix_fmts),
                              &avctx->outputs[0]->incfg.formats)) < 0)
        return err;
    return 0;
}

int amf_copy_surface(AVFilterContext *avctx, const AVFrame *frame,
    AMFSurface* surface)
{
    AMFPlane *plane;
    uint8_t  *dst_data[4];
    int       dst_linesize[4];
    int       planes;
    int       i;

    planes = surface->pVtbl->GetPlanesCount(surface);
    av_assert0(planes < FF_ARRAY_ELEMS(dst_data));

    for (i = 0; i < planes; i++) {
        plane = surface->pVtbl->GetPlaneAt(surface, i);
        dst_data[i] = plane->pVtbl->GetNative(plane);
        dst_linesize[i] = plane->pVtbl->GetHPitch(plane);
    }
    av_image_copy(dst_data, dst_linesize,
        (const uint8_t**)frame->data, frame->linesize, frame->format,
        frame->width, frame->height);

    return 0;
}

int amf_init_scale_config(AVFilterLink *outlink, enum AVPixelFormat *in_format)
{
    AVFilterContext *avctx = outlink->src;
    AVFilterLink   *inlink = avctx->inputs[0];
    AMFScaleContext  *ctx = avctx->priv;
    AVHWFramesContext *hwframes_out;
    int err;
    AMF_RESULT res;

    if ((err = ff_scale_eval_dimensions(avctx,
                                        ctx->w_expr, ctx->h_expr,
                                        inlink, outlink,
                                        &ctx->width, &ctx->height)) < 0)
        return err;

    ff_scale_adjust_dimensions(inlink, &ctx->width, &ctx->height,
                               ctx->force_original_aspect_ratio, ctx->force_divisible_by);

    av_buffer_unref(&ctx->amf_device_ref);
    av_buffer_unref(&ctx->hwframes_in_ref);
    av_buffer_unref(&ctx->hwframes_out_ref);

    if (inlink->hw_frames_ctx) {
        AVHWFramesContext *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
        if (frames_ctx->device_ctx->type == AV_HWDEVICE_TYPE_AMF) {
            AVAMFDeviceContext * amf_ctx =  frames_ctx->device_ctx->hwctx;
            ctx->amf_device_ctx_internal = av_buffer_ref(amf_ctx->internal);
        }
        if (av_amf_av_to_amf_format(frames_ctx->sw_format) == AMF_SURFACE_UNKNOWN) {
            av_log(avctx, AV_LOG_ERROR, "Format of input frames context (%s) is not supported by AMF.\n",
                   av_get_pix_fmt_name(frames_ctx->sw_format));
            return AVERROR(EINVAL);
        }

        err = av_hwdevice_ctx_create_derived(&ctx->amf_device_ref, AV_HWDEVICE_TYPE_AMF, frames_ctx->device_ref, 0);
        if (err < 0)
            return err;

        ctx->hwframes_in_ref = av_buffer_ref(inlink->hw_frames_ctx);
        if (!ctx->hwframes_in_ref)
            return AVERROR(ENOMEM);

        ctx->hwframes_out_ref = av_hwframe_ctx_alloc(frames_ctx->device_ref);
        if (!ctx->hwframes_out_ref)
            return AVERROR(ENOMEM);

        hwframes_out = (AVHWFramesContext*)ctx->hwframes_out_ref->data;
        hwframes_out->format    = outlink->format;
        hwframes_out->sw_format = frames_ctx->sw_format;
    } else if (avctx->hw_device_ctx) {
        AVHWDeviceContext   *hwdev_ctx;
        err = av_hwdevice_ctx_create_derived(&ctx->amf_device_ref, AV_HWDEVICE_TYPE_AMF, avctx->hw_device_ctx, 0);
        if (err < 0)
            return err;
        hwdev_ctx = (AVHWDeviceContext*)avctx->hw_device_ctx->data;
        if (hwdev_ctx->type == AV_HWDEVICE_TYPE_AMF)
        {
            AVAMFDeviceContext * amf_ctx =  hwdev_ctx->hwctx;
            ctx->amf_device_ctx_internal = av_buffer_ref(amf_ctx->internal);
        }
        ctx->hwdevice_ref = av_buffer_ref(avctx->hw_device_ctx);
        if (!ctx->hwdevice_ref)
            return AVERROR(ENOMEM);

        ctx->hwframes_out_ref = av_hwframe_ctx_alloc(ctx->hwdevice_ref);
        if (!ctx->hwframes_out_ref)
            return AVERROR(ENOMEM);

        hwframes_out = (AVHWFramesContext*)ctx->hwframes_out_ref->data;
        hwframes_out->format    = AV_PIX_FMT_AMF;
        hwframes_out->sw_format = outlink->format;
    } else {
        AVAMFDeviceContextInternal *wrapped = av_mallocz(sizeof(*wrapped));
        ctx->amf_device_ctx_internal = av_buffer_create((uint8_t *)wrapped, sizeof(*wrapped),
                                                av_amf_context_internal_free, NULL, 0);
        if ((res == av_amf_context_internal_create((AVAMFDeviceContextInternal *)ctx->amf_device_ctx_internal->data, avctx, "", NULL, 0)) != 0) {
            return res;
        }
        ctx->hwframes_out_ref = av_hwframe_ctx_alloc(ctx->amf_device_ref);
        if (!ctx->hwframes_out_ref)
            return AVERROR(ENOMEM);

        hwframes_out = (AVHWFramesContext*)ctx->hwframes_out_ref->data;
        hwframes_out->format    = outlink->format;
        hwframes_out->sw_format = inlink->format;
    }

    if (ctx->format != AV_PIX_FMT_NONE) {
        hwframes_out->sw_format = ctx->format;
    }

    if (inlink->format == AV_PIX_FMT_AMF) {
        if (!inlink->hw_frames_ctx || !inlink->hw_frames_ctx->data)
            return AVERROR(EINVAL);
        else
            *in_format = ((AVHWFramesContext*)inlink->hw_frames_ctx->data)->sw_format;
    } else
        *in_format = inlink->format;

    outlink->w = ctx->width;
    outlink->h = ctx->height;

    hwframes_out->width = outlink->w;
    hwframes_out->height = outlink->h;

    err = av_hwframe_ctx_init(ctx->hwframes_out_ref);
    if (err < 0)
        return err;

    outlink->hw_frames_ctx = av_buffer_ref(ctx->hwframes_out_ref);
    if (!outlink->hw_frames_ctx) {
        return AVERROR(ENOMEM);
    }
    return 0;
}

void amf_free_amfsurface(void *opaque, uint8_t *data)
{
    AVFilterContext *avctx = (AVFilterContext *)opaque;
    AMFSurface *surface = (AMFSurface*)(opaque);
    // FIXME: release surface properly
    int count = surface->pVtbl->Release(surface);
}

AVFrame *amf_amfsurface_to_avframe(AVFilterContext *avctx, AMFSurface* pSurface)
{
    AVFrame *frame = av_frame_alloc();
    AMFScaleContext  *ctx = avctx->priv;

    if (!frame)
        return NULL;

    if (ctx->hwframes_out_ref) {
        AVHWFramesContext *hwframes_out = (AVHWFramesContext *)ctx->hwframes_out_ref->data;
        if (hwframes_out->format == AV_PIX_FMT_AMF) {
            int ret = av_hwframe_get_buffer(ctx->hwframes_out_ref, frame, 0);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "Get hw frame failed.\n");
                av_frame_free(&frame);
                return ret;
            }
            frame->data[3] = pSurface;
            frame->buf[1] = av_buffer_create((uint8_t *)pSurface, sizeof(AMFSurface),
                                            amf_free_amfsurface,
                                            (void*)avctx,
                                            AV_BUFFER_FLAG_READONLY);
        } else { // FIXME: add processing of other hw formats
            av_log(ctx, AV_LOG_ERROR, "Unknown pixel format\n");
            return NULL;
        }
    } else {

        switch (pSurface->pVtbl->GetMemoryType(pSurface))
        {
    #if CONFIG_D3D11VA
            case AMF_MEMORY_DX11:
            {
                AMFPlane *plane0 = pSurface->pVtbl->GetPlaneAt(pSurface, 0);
                frame->data[0] = plane0->pVtbl->GetNative(plane0);
                frame->data[1] = (uint8_t*)(intptr_t)0;

                frame->buf[0] = av_buffer_create(NULL,
                                        0,
                                        amf_free_amfsurface,
                                        pSurface,
                                        AV_BUFFER_FLAG_READONLY);
            }
            break;
    #endif
    #if CONFIG_DXVA2
            case AMF_MEMORY_DX9:
            {
                AMFPlane *plane0 = pSurface->pVtbl->GetPlaneAt(pSurface, 0);
                frame->data[3] = plane0->pVtbl->GetNative(plane0);

                frame->buf[0] = av_buffer_create(NULL,
                                        0,
                                        amf_free_amfsurface,
                                        pSurface,
                                        AV_BUFFER_FLAG_READONLY);
            }
            break;
    #endif
        default:
            // FIXME: add support for other memory types
            {
                av_log(avctx, AV_LOG_ERROR, "Unsupported memory type : %d\n", pSurface->pVtbl->GetMemoryType(pSurface));
                return NULL;
            }
        }
    }

    return frame;
}

int amf_avframe_to_amfsurface(AVFilterContext *avctx, const AVFrame *frame, AMFSurface** ppSurface)
{
    AMFScaleContext *ctx = avctx->priv;
    AVAMFDeviceContextInternal* internal = (AVAMFDeviceContextInternal *)ctx->amf_device_ctx_internal->data;
    AMFSurface *surface;
    AMF_RESULT  res;
    int hw_surface = 0;

    switch (frame->format) {
#if CONFIG_D3D11VA
    case AV_PIX_FMT_D3D11:
        {
            static const GUID AMFTextureArrayIndexGUID = { 0x28115527, 0xe7c3, 0x4b66, { 0x99, 0xd3, 0x4f, 0x2a, 0xe6, 0xb4, 0x7f, 0xaf } };
            ID3D11Texture2D *texture = (ID3D11Texture2D*)frame->data[0]; // actual texture
            int index = (intptr_t)frame->data[1]; // index is a slice in texture array is - set to tell AMF which slice to use
            texture->lpVtbl->SetPrivateData(texture, &AMFTextureArrayIndexGUID, sizeof(index), &index);

            res = internal->context->pVtbl->CreateSurfaceFromDX11Native(internal->context, texture, &surface, NULL); // wrap to AMF surface
            AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(ENOMEM), "CreateSurfaceFromDX11Native() failed  with error %d\n", res);
            hw_surface = 1;
        }
        break;
#endif
    // FIXME: need to use hw_frames_ctx to get texture
    case AV_PIX_FMT_AMF:
        {
            surface = (AMFSurface*)frame->data[3]; // actual surface
            hw_surface = 1;
        }
        break;

#if CONFIG_DXVA2
    case AV_PIX_FMT_DXVA2_VLD:
        {
            IDirect3DSurface9 *texture = (IDirect3DSurface9 *)frame->data[3]; // actual texture

            res = internal->context->pVtbl->CreateSurfaceFromDX9Native(internal->context, texture, &surface, NULL); // wrap to AMF surface
            AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(ENOMEM), "CreateSurfaceFromDX9Native() failed  with error %d\n", res);
            hw_surface = 1;
        }
        break;
#endif
    default:
        {
            AMF_SURFACE_FORMAT amf_fmt = av_amf_av_to_amf_format(frame->format);
            res = internal->context->pVtbl->AllocSurface(internal->context, AMF_MEMORY_HOST, amf_fmt, frame->width, frame->height, &surface);
            AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(ENOMEM), "AllocSurface() failed  with error %d\n", res);
            amf_copy_surface(avctx, frame, surface);
        }
        break;
    }

    if (hw_surface) {
        // input HW surfaces can be vertically aligned by 16; tell AMF the real size
        surface->pVtbl->SetCrop(surface, 0, 0, frame->width, frame->height);
    }

    surface->pVtbl->SetPts(surface, frame->pts);
    *ppSurface = surface;
    return 0;
}
