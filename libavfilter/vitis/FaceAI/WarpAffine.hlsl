// WarpAffine.hlsl
cbuffer ConstantBuffer : register(b0)
{
    float width;
    float height;
    float flags;
    float borderMode;
    float4 borderValue;
    float3x2 affineMatrix;
    float srcwidth;
    float srcheight;
};

Texture2D<float4> srcImage : register(t0);
RWTexture2D<float4> dstImage : register(u0);

SamplerState samLinear : register(s0);

float4 sampleBilinear(Texture2D<float4> srcimg, float2 texCoord)
{
    float2 texSize;
    srcimg.GetDimensions(texSize.x, texSize.y);

    float2 texelSize = 1.0 / texSize;
    float2 texelCoord = texCoord * texSize - 0.5;
    float2 blend = frac(texelCoord);
    texelCoord = floor(texelCoord);

    float4 c00 = srcimg.SampleLevel(samLinear, texelCoord * texelSize, 0);
    float4 c10 = srcimg.SampleLevel(samLinear, (texelCoord + float2(1, 0)) * texelSize, 0);
    float4 c01 = srcimg.SampleLevel(samLinear, (texelCoord + float2(0, 1)) * texelSize, 0);
    float4 c11 = srcimg.SampleLevel(samLinear, (texelCoord + float2(1, 1)) * texelSize, 0);

    return lerp(lerp(c00, c10, blend.x), lerp(c01, c11, blend.x), blend.y);
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    //dstImage[DTid.xy] = affineMatrix._31_12_22_32;
    //dstImage[DTid.xy] = affineMatrix._11_21_31_12;
    //dstImage[DTid.xy] = float4(128,512,srcwidth,srcheight);
    //return;
    if (DTid.x >= width || DTid.y >= height)
    {
        return;
    }
        

    float3 pos = float3(DTid.x, DTid.y, 1.0);
    float2 newPos = mul(pos, affineMatrix);
    //newPos = newPos + {width/2,height/2};
    //newPos.x = DTid.x * affineMatrix._11 + DTid.y * affineMatrix._21 + affineMatrix._31;
    //newPos.y = DTid.x * affineMatrix._12 + DTid.y * affineMatrix._22 + affineMatrix._32;
    
    float4 color;

    if (newPos.x < 0 || newPos.y < 0 || newPos.x >= srcwidth || newPos.y >= srcheight)
    {
        if (borderMode == 0) // BORDER_CONSTANT
        {
            color = borderValue;
        }
        else if (borderMode == 1) // BORDER_REPLICATE
        {
            newPos.x = clamp(newPos.x, 0, srcwidth - 1);
            newPos.y = clamp(newPos.y, 0, srcheight - 1);
            color = srcImage.SampleLevel(samLinear, newPos, 0);
        }
    }
    else
    {
        if (flags == 0) // INTER_NEAREST
        {
            newPos = floor(newPos + 0.5);
            color = srcImage.SampleLevel(samLinear, newPos / float2(srcwidth, srcheight), 0);
        }
        else if (flags == 1) // INTER_LINEAR
        {
            color = sampleBilinear(srcImage, newPos / float2(srcwidth, srcheight));
        }
        // Add other interpolation methods if needed
    }

    if (color.a < 0)
        color.a = 0;
    if (color.a > 1)
        color.a = 1;

    dstImage[DTid.xy] = color;
}
