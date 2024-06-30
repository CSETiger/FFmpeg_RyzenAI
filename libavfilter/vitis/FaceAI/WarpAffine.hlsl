// WarpAffine.hlsl
cbuffer ConstantBuffer : register(b0)
{
    float3x2 affineMatrix;
    uint width;
    uint height;
    uint flags;
    uint borderMode;
    float4 borderValue;
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

[numthreads(16, 16, 4)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // dstImage[DTid.xy] =float4(0.5,0.5,0.5,0.5);

    // return;

    if (DTid.x >= width || DTid.y >= height)
    {
        //dstImage[DTid.xy] =float4(1.0,0.5,1.5,0.5);
        //dstImage[DTid.xy] = float4(width,height,x,y)
        return;
    }
        

    float3 pos = float3(DTid.xy, 1.0);
    float2 newPos = mul(pos, affineMatrix);
    
    float4 color;

    if (newPos.x < 0 || newPos.y < 0 || newPos.x >= width || newPos.y >= height)
    {
        if (borderMode == 0) // BORDER_CONSTANT
        {
            color = borderValue;
        }
        else if (borderMode == 1) // BORDER_REPLICATE
        {
            newPos.x = clamp(newPos.x, 0, width - 1);
            newPos.y = clamp(newPos.y, 0, height - 1);
            color = srcImage.SampleLevel(samLinear, newPos, 0);
        }
    }
    else
    {
        if (flags == 0) // INTER_NEAREST
        {
            newPos = floor(newPos + 0.5);
            color = srcImage.SampleLevel(samLinear, newPos / float2(width, height), 0);
        }
        else if (flags == 1) // INTER_LINEAR
        {
            color = sampleBilinear(srcImage, newPos / float2(width, height));
        }
        // Add other interpolation methods if needed
    }

    dstImage[DTid.xy] = color;
}
