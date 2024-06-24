cbuffer Constants : register(b0)
{
    float3x3 affineMatrix;
    float width;
    float height;
    float flags;
    float borderMode;
    float3 borderValue;
};

Texture2D<float4> inputImage : register(t0);
RWTexture2D<float4> outputImage : register(u0);

float4 getBorderValue(float3 borderValue, float borderMode)
{
    if (borderMode == 0) // BORDER_CONSTANT
    {
        return float4(borderValue, 1.0);
    }
    // Other border modes can be implemented here
    return float4(0, 0, 0, 0);
}

float2 replicateBorder(float2 pos, float width, float height)
{
    pos.x = clamp(pos.x, 0.0, width - 1.0);
    pos.y = clamp(pos.y, 0.0, height - 1.0);
    return pos;
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int x = DTid.x;
    int y = DTid.y;

    if (x >= width || y >= height)
        return;

    float3 pos = float3(x, y, 1);
    float3 newPos = mul(affineMatrix, pos);
    float2 pos2D = newPos.xy;

    float4 color;

    if (newPos.x < 0 || newPos.x >= width || newPos.y < 0 || newPos.y >= height)
    {
        if (borderMode == 1) // BORDER_REPLICATE
        {
            pos2D = replicateBorder(pos2D, width, height);
            color = inputImage.Load(int3(pos2D, 0));
        }
        else
        {
            color = getBorderValue(borderValue, borderMode);
        }
    }
    else
    {
        if (flags == 0) // INTER_NEAREST
        {
            int2 nearestPos = int2(round(pos2D));
            color = inputImage.Load(int3(nearestPos, 0));
        }
        else if (flags == 1) // INTER_LINEAR
        {
            float2 frac = float2(pos2D);
            int2 p1 = int2(floor(pos2D));
            int2 p2 = p1 + int2(1, 0);
            int2 p3 = p1 + int2(0, 1);
            int2 p4 = p1 + int2(1, 1);

            float4 c1 = inputImage.Load(int3(p1, 0));
            float4 c2 = inputImage.Load(int3(p2, 0));
            float4 c3 = inputImage.Load(int3(p3, 0));
            float4 c4 = inputImage.Load(int3(p4, 0));

            color = lerp(lerp(c1, c2, frac.x), lerp(c3, c4, frac.x), frac.y);
        }
        // Additional interpolation methods can be implemented here
    }

    outputImage[DTid.xy] = color;
}