#include "Msnhnet/cv/MsnhCVVideo.h"

#define PL_MPEG_IMPLEMENTATION
#include "../3rdparty/pl_mpeg/pl_mpeg.h"

#include "../3rdparty/stb/stb_image_write.h"

#include "../3rdparty/stb/stb_image.h"

namespace Msnhnet
{
#define ZEROIZE(x) {memset(&x, 0, sizeof(x));}

AviEncoder::AviEncoder(const char *filename, const uint32_t &width, const uint32_t &height, const uint32_t &bpp, const char *fourcc, const uint32_t &fps, AviEncoder::AviAudio *audio)
{
    ZEROIZE(aviHeader);
    ZEROIZE(streamHeaderV);
    ZEROIZE(streamFormatV);
    ZEROIZE(streamHeaderA);
    ZEROIZE(streamFormatA);

    outFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    if (checkFourcc(fourcc) != 0)
        throw Exception(1, "[Avi Encoder]: fourcc not valid! \n",__FILE__,__LINE__,__FUNCTION__);
    if (fps < 1)
        throw Exception(1, "[Avi Encoder]: fps should >= 1! \n",__FILE__,__LINE__,__FUNCTION__);

    try
    {
        outFile.open(filename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

        aviHeader.timeDelay = 1000000 / fps;
        aviHeader.dataRate = width * height * bpp / 8;
        aviHeader.flags = 0x10;

        if (audio)
        {
            aviHeader.dataStreams = 2;
        }
        else
        {
            aviHeader.dataStreams = 1;
        }

        aviHeader.numberOfFrames = 0;
        aviHeader.width = width;
        aviHeader.height = height;
        aviHeader.bufferSize = (width * height * bpp / 8);

        (void) strcpy(streamHeaderV.data_type, "vids");
        (void) memcpy(streamHeaderV.codec, fourcc, 4);
        streamHeaderV.timeScale = 1;
        streamHeaderV.dataRate = fps;
        streamHeaderV.bufferSize = (width * height * bpp / 8);
        streamHeaderV.dataLength = 0;

        streamFormatV.headerSize = 40;
        streamFormatV.width = width;
        streamFormatV.height = height;
        streamFormatV.numPlanes = 1;
        streamFormatV.bitsPerPixel = bpp;
        streamFormatV.compressionType = (static_cast<uint32_t>(fourcc[3]) << 24) + (static_cast<uint32_t>( fourcc[2]) << 16)
                + (static_cast<uint32_t>( fourcc[1]) << 8) + (static_cast<uint32_t>( fourcc[0]));
        streamFormatV.imageSize = width * height * 3;
        streamFormatV.colorsUsed = 0;
        streamFormatV.colorsImportant = 0;

        streamFormatV.palette = NULL;
        streamFormatV.paletteCount = 0;

        if (audio)
        {

            memcpy(streamHeaderA.data_type, "auds", 4);
            streamHeaderA.codec[0] = 1;
            streamHeaderA.codec[1] = 0;
            streamHeaderA.codec[2] = 0;
            streamHeaderA.codec[3] = 0;
            streamHeaderA.timeScale = 1;
            streamHeaderA.dataRate = audio->samplesPerSecond;
            streamHeaderA.bufferSize = audio->channels * (audio->bits / 8) * audio->samplesPerSecond;

            streamHeaderA.audioQuality = -1;
            streamHeaderA.sampleSize = (audio->bits / 8) * audio->channels;

            streamFormatA.formatType = 1;
            streamFormatA.channels = audio->channels;
            streamFormatA.sampleRate = audio->samplesPerSecond;
            streamFormatA.bytesPerSecond = audio->channels * (audio->bits / 8) * audio->samplesPerSecond;
            streamFormatA.blockAlign = audio->channels * (audio->bits / 8);
            streamFormatA.bitsPerSample = audio->bits;
            streamFormatA.size = 0;
        }

        writeCharsBin("RIFF", 4);
        writeInt(0);
        writeCharsBin("AVI ", 4);

        writeAviHeaderChunk();

        writeCharsBin("LIST", 4);

        marker = static_cast<uint32_t>(outFile.tellp());

        writeInt(0);
        writeCharsBin("movi", 4);

        offsetsLen = 1024;
        offsets = new uint32_t[offsetsLen];
        offsetsPtr = 0;

    }
    catch (...)
    {
        if (outFile.is_open())
        {
            outFile.close();
        }
        throw Exception(1, "[Avi Encoder]: Init error occured! \n",__FILE__,__LINE__,__FUNCTION__);
    }
}

AviEncoder::~AviEncoder()
{
    if (outFile.is_open())
    {
        outFile.close();
    }

    delete[] offsets;
    offsets = nullptr;
}

void AviEncoder::addVideoFrame(const char *buffer, const size_t &len)
{
    size_t maxiPad; 

    size_t t;

    if (!buffer)
    {
        throw Exception(1,"[Avi Encoder]: Buffer can't be null! \n",__FILE__,__LINE__,__FUNCTION__);
    }
    try
    {
        offsetCount++;
        streamHeaderV.dataLength++;

        maxiPad = len % 4;
        if (maxiPad > 0)
        {
            maxiPad = 4 - maxiPad;
        }

        if (offsetCount >= offsetsLen)
        {
            offsetsLen += 1024;
            delete[] offsets;
            offsets = new uint32_t[offsetsLen];
        }

        offsets[offsetsPtr++] = static_cast<uint32_t>(len + maxiPad);

        writeCharsBin("00dc", 4);

        writeInt(static_cast<uint32_t>(len + maxiPad));

        outFile.write(buffer, len);

        for (t = 0; t < maxiPad; t++)
            outFile.write("\0", 1);

    }
    catch (std::system_error& e)
    {
        throw Exception(1,"[Avi Encoder]: " + std::string(e.what()) + "\n",__FILE__,__LINE__,__FUNCTION__);
    }

}

void AviEncoder::addAudioFrame(const char *buffer, const size_t &len)
{
    size_t maxiPad; 

    size_t t;

    if (!buffer)
    {
        throw Exception(1,"[Avi Encoder]: Buffer can't be null! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    try
    {
        offsetCount++;

        maxiPad = len % 4;
        if (maxiPad > 0)
            maxiPad = 4 - maxiPad;

        if (offsetCount >= offsetsLen) {
            offsetsLen += 1024;
            delete[] offsets;
            offsets = new uint32_t[offsetsLen];
        }

        offsets[offsetsPtr++] = static_cast<uint32_t>((len + maxiPad) | 0x80000000);

        writeCharsBin("01wb", 4);
        writeInt(static_cast<uint32_t>(len + maxiPad));

        outFile.write((char *) buffer, len);

        for (t = 0; t < maxiPad; t++)
            outFile.write("\0", 1);

        streamHeaderA.dataLength += static_cast<uint32_t>(len + maxiPad);

    }
    catch (std::system_error& e)
    {
        throw Exception(1,"[Avi Encoder]: " + std::string(e.what()) + "\n",__FILE__,__LINE__,__FUNCTION__);
    }

}

void AviEncoder::finalize()
{
    uint32_t t;
    try
    {
        t = static_cast<uint32_t>(outFile.tellp());
        outFile.seekp(marker, std::ios_base::beg);
        writeInt(static_cast<uint32_t>(t - marker - 4));
        outFile.seekp(t, std::ios_base::beg);

        writeIndex(offsetCount, offsets);

        delete[] offsets;
        offsets = NULL;

        aviHeader.numberOfFrames = streamHeaderV.dataLength;

        t = static_cast<uint32_t>(outFile.tellp());
        outFile.seekp(12, std::ios_base::beg);
        writeAviHeaderChunk();
        outFile.seekp(t, std::ios_base::beg);

        t = static_cast<uint32_t>(outFile.tellp());
        outFile.seekp(4, std::ios_base::beg);
        writeInt(static_cast<uint32_t>(t - 8));
        outFile.seekp(t, std::ios_base::beg);

        if (streamFormatV.palette) 

            delete[] streamFormatV.palette;

        outFile.close();
    }
    catch (std::system_error& e)
    {
        throw Exception(1,"[Avi Encoder]: " + std::string(e.what()) + "\n",__FILE__,__LINE__,__FUNCTION__);
    }
}

void AviEncoder::setFramerate(const uint32_t &fps)
{
    streamHeaderV.dataRate = fps;
    aviHeader.timeDelay = (10000000 / fps);
}

void AviEncoder::setFourccCodec(const char *fourcc)
{
    if (checkFourcc(fourcc) != 0)
    {
        throw Exception(1, "[Avi Encoder]: fourcc not valid! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    memcpy(streamHeaderV.codec, fourcc, 4);
    streamFormatV.compressionType = (static_cast<uint32_t>( fourcc[3]) << 24) + (static_cast<uint32_t>( fourcc[2]) << 16)
            + (static_cast<uint32_t>( fourcc[1]) << 8) + (static_cast<uint32_t>( fourcc[0]));
}

void AviEncoder::setVideoFrameSize(const uint32_t &width, const uint32_t &height)
{
    uint32_t size = (width * height * 3);

    aviHeader.dataRate = size;
    aviHeader.width = width;
    aviHeader.height = height;
    aviHeader.bufferSize = size;
    streamHeaderV.bufferSize = size;
    streamFormatV.width = width;
    streamFormatV.height = height;
    streamFormatV.imageSize = size;
}

void AviEncoder::writeAviHeader(AviEncoder::AviHeader *aviHeader)
{
    uint32_t marker, t;

    writeCharsBin("avih", 4);
    marker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);

    writeInt(aviHeader->timeDelay);
    writeInt(aviHeader->dataRate);
    writeInt(aviHeader->reserved);

    writeInt(aviHeader->flags);

    writeInt(aviHeader->numberOfFrames);
    writeInt(aviHeader->initialFrames);
    writeInt(aviHeader->dataStreams);
    writeInt(aviHeader->bufferSize);
    writeInt(aviHeader->width);
    writeInt(aviHeader->height);
    writeInt(aviHeader->timeScale);
    writeInt(aviHeader->playbackDataRate);
    writeInt(aviHeader->startingTime);
    writeInt(aviHeader->dataLength);

    t = static_cast<uint32_t>(outFile.tellp());
    outFile.seekp(marker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - marker - 4));
    outFile.seekp(t, std::ios_base::beg);
}

void AviEncoder::writeStreamHeader(AviEncoder::AviStreamHeader *streamHeader)
{
    uint32_t marker, t;

    writeCharsBin("strh", 4);
    marker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);

    writeCharsBin(streamHeader->data_type, 4);
    writeCharsBin(streamHeader->codec, 4);
    writeInt(streamHeader->flags);
    writeInt(streamHeader->priority);
    writeInt(streamHeader->initialFrames);
    writeInt(streamHeader->timeScale);
    writeInt(streamHeader->dataRate);
    writeInt(streamHeader->startTime);
    writeInt(streamHeader->dataLength);
    writeInt(streamHeader->bufferSize);
    writeInt(streamHeader->videoQuality);
    writeInt(streamHeader->sampleSize);
    writeInt(0);
    writeInt(0);

    t = static_cast<uint32_t>(outFile.tellp());
    outFile.seekp(marker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - marker - 4));
    outFile.seekp(t, std::ios_base::beg);
}

void AviEncoder::writeStreamFormatV(AviEncoder::AviStreamFormatV *streamFormatV)
{
    uint32_t marker, t;
    uint32_t i;

    writeCharsBin("strf", 4);
    marker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);
    writeInt(streamFormatV->headerSize);
    writeInt(streamFormatV->width);
    writeInt(streamFormatV->height);
    writeuint16_t(streamFormatV->numPlanes);
    writeuint16_t(streamFormatV->bitsPerPixel);
    writeInt(streamFormatV->compressionType);
    writeInt(streamFormatV->imageSize);
    writeInt(streamFormatV->xPelsPerMeter);
    writeInt(streamFormatV->yPelsPerMeter);
    writeInt(streamFormatV->colorsUsed);
    writeInt(streamFormatV->colorsImportant);

    if (streamFormatV->colorsUsed != 0)
    {
        for (i = 0; i < streamFormatV->colorsUsed; i++)
        {
            unsigned char c = streamFormatV->palette[i] & 255;
            outFile.write((char *) &c, 1);
            c = (streamFormatV->palette[i] >> 8) & 255;
            outFile.write((char *) &c, 1);
            c = (streamFormatV->palette[i] >> 16) & 255;
            outFile.write((char *) &c, 1);
            outFile.write("\0", 1);
        }
    }

    t = static_cast<uint32_t>(outFile.tellp());
    outFile.seekp(marker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - marker - 4));
    outFile.seekp(t, std::ios_base::beg);
}

void AviEncoder::writeStreamFormatA(AviEncoder::AviStreamFormatA *streamFormatA)
{
    uint32_t marker, t;

    writeCharsBin("strf", 4);
    marker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);
    writeuint16_t(streamFormatA->formatType);
    writeuint16_t(streamFormatA->channels);
    writeInt(streamFormatA->sampleRate);
    writeInt(streamFormatA->bytesPerSecond);
    writeuint16_t(streamFormatA->blockAlign);
    writeuint16_t(streamFormatA->bitsPerSample);
    writeuint16_t(streamFormatA->size);

    t = static_cast<uint32_t>(outFile.tellp());
    outFile.seekp(marker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - marker - 4));
    outFile.seekp(t, std::ios_base::beg);
}

void AviEncoder::writeAviHeaderChunk()
{
    uint32_t marker, t;
    uint32_t subMarker;

    writeCharsBin("LIST", 4);
    marker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);
    writeCharsBin("hdrl", 4);
    writeAviHeader(&aviHeader);

    writeCharsBin("LIST", 4);
    subMarker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);
    writeCharsBin("strl", 4);
    writeStreamHeader(&streamHeaderV);
    writeStreamFormatV(&streamFormatV);

    t = static_cast<uint32_t>(outFile.tellp());

    outFile.seekp(subMarker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - subMarker - 4));
    outFile.seekp(t, std::ios_base::beg);

    if (aviHeader.dataStreams == 2)
    {
        writeCharsBin("LIST", 4);
        subMarker = static_cast<uint32_t>(outFile.tellp());
        writeInt(0);
        writeCharsBin("strl", 4);
        writeStreamHeader(&streamHeaderA);
        writeStreamFormatA(&streamFormatA);

        t = static_cast<uint32_t>(outFile.tellp());
        outFile.seekp(subMarker, std::ios_base::beg);
        writeInt(static_cast<uint32_t>(t - subMarker - 4));
        outFile.seekp(t, std::ios_base::beg);
    }

    t = static_cast<uint32_t>(outFile.tellp());
    outFile.seekp(marker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - marker - 4));
    outFile.seekp(t, std::ios_base::beg);
}

void AviEncoder::writeIndex(const uint32_t &count, uint32_t *offsets)
{
    uint32_t marker = 0;
    uint32_t t      = 0;
    uint32_t offset = 4;

    if (!offsets)
        throw 1;

    writeCharsBin("idx1", 4);
    marker = static_cast<uint32_t>(outFile.tellp());
    writeInt(0);

    for (t = 0; t < count; t++)
    {
        if ((offsets[t] & 0x80000000) == 0)
        {
            writeChars("00dc");
        }
        else
        {
            writeChars("01wb");
            offsets[t] &= 0x7fffffff;
        }
        writeInt(0x10);
        writeInt(offset);
        writeInt(offsets[t]);

        offset = offset + offsets[t] + 8;
    }

    t = static_cast<uint32_t>(outFile.tellp());
    outFile.seekp(marker, std::ios_base::beg);
    writeInt(static_cast<uint32_t>(t - marker - 4));
    outFile.seekp(t, std::ios_base::beg);
}

int AviEncoder::checkFourcc(const char *fourcc)
{
    int ret = 0;

    const char valid_fourcc[] = "3IV1 3IV2 8BPS"
                                "AASC ABYR ADV1 ADVJ AEMI AFLC AFLI AJPG AMPG ANIM AP41 ASLC"
                                "ASV1 ASV2 ASVX AUR2 AURA AVC1 AVRN"
                                "BA81 BINK BLZ0 BT20 BTCV BW10 BYR1 BYR2"
                                "CC12 CDVC CFCC CGDI CHAM CJPG CMYK CPLA CRAM CSCD CTRX CVID"
                                "CWLT CXY1 CXY2 CYUV CYUY"
                                "D261 D263 DAVC DCL1 DCL2 DCL3 DCL4 DCL5 DIV3 DIV4 DIV5 DIVX"
                                "DM4V DMB1 DMB2 DMK2 DSVD DUCK DV25 DV50 DVAN DVCS DVE2 DVH1"
                                "DVHD DVSD DVSL DVX1 DVX2 DVX3 DX50 DXGM DXTC DXTN"
                                "EKQ0 ELK0 EM2V ES07 ESCP ETV1 ETV2 ETVC"
                                "FFV1 FLJP FMP4 FMVC FPS1 FRWA FRWD FVF1"
                                "GEOX GJPG GLZW GPEG GWLT"
                                "H260 H261 H262 H263 H264 H265 H266 H267 H268 H269"
                                "HDYC HFYU HMCR HMRR"
                                "I263 ICLB IGOR IJPG ILVC ILVR IPDV IR21 IRAW ISME"
                                "IV30 IV31 IV32 IV33 IV34 IV35 IV36 IV37 IV38 IV39 IV40 IV41"
                                "IV41 IV43 IV44 IV45 IV46 IV47 IV48 IV49 IV50"
                                "JBYR JPEG JPGL"
                                "KMVC"
                                "L261 L263 LBYR LCMW LCW2 LEAD LGRY LJ11 LJ22 LJ2K LJ44 LJPG"
                                "LMP2 LMP4 LSVC LSVM LSVX LZO1"
                                "M261 M263 M4CC M4S2 MC12 MCAM MJ2C MJPG MMES MP2A MP2T MP2V"
                                "MP42 MP43 MP4A MP4S MP4T MP4V MPEG MPNG MPG4 MPGI MR16 MRCA MRLE"
                                "MSVC MSZH"
                                "MTX1 MTX2 MTX3 MTX4 MTX5 MTX6 MTX7 MTX8 MTX9"
                                "MVI1 MVI2 MWV1"
                                "NAVI NDSC NDSM NDSP NDSS NDXC NDXH NDXP NDXS NHVU NTN1 NTN2"
                                "NVDS NVHS"
                                "NVS0 NVS1 NVS2 NVS3 NVS4 NVS5"
                                "NVT0 NVT1 NVT2 NVT3 NVT4 NVT5"
                                "PDVC PGVV PHMO PIM1 PIM2 PIMJ PIXL PJPG PVEZ PVMM PVW2"
                                "QPEG QPEQ"
                                "RGBT RLE RLE4 RLE8 RMP4 RPZA RT21 RV20 RV30 RV40 S422 SAN3"
                                "SDCC SEDG SFMC SMP4 SMSC SMSD SMSV SP40 SP44 SP54 SPIG SQZ2"
                                "STVA STVB STVC STVX STVY SV10 SVQ1 SVQ3"
                                "TLMS TLST TM20 TM2X TMIC TMOT TR20 TSCC TV10 TVJP TVMJ TY0N"
                                "TY2C TY2N"
                                "UCOD ULTI"
                                "V210 V261 V655 VCR1 VCR2 VCR3 VCR4 VCR5 VCR6 VCR7 VCR8 VCR9"
                                "VDCT VDOM VDTZ VGPX VIDS VIFP VIVO VIXL VLV1 VP30 VP31 VP40"
                                "VP50 VP60 VP61 VP62 VP70 VP80 VQC1 VQC2 VQJC VSSV VUUU VX1K"
                                "VX2K VXSP VYU9 VYUY"
                                "WBVC WHAM WINX WJPG WMV1 WMV2 WMV3 WMVA WNV1 WVC1"
                                "X263 X264 XLV0 XMPG XVID"
                                "XWV0 XWV1 XWV2 XWV3 XWV4 XWV5 XWV6 XWV7 XWV8 XWV9"
                                "XXAN"
                                "Y16 Y411 Y41P Y444 Y8 YC12 YUV8 YUV9 YUVP YUY2 YUYV YV12 YV16"
                                "YV92"
                                "ZLIB ZMBV ZPEG ZYGO ZYYY";

    if (!fourcc)
    {
        (void) fputs("fourcc cannot be NULL", stderr);
        return -1;
    }

    if (strchr(fourcc, ' ') || !strstr(valid_fourcc, fourcc))
    {
        ret = 1;
    }
    return ret;
}

void AviEncoder::writeInt(uint32_t n)
{
    unsigned char buffer[4];

    buffer[0] = n;
    buffer[1] = n >> 8;
    buffer[2] = n >> 16;
    buffer[3] = n >> 24;

    outFile.write((char *) buffer, 4);
}

void AviEncoder::writeuint16_t(uint16_t n)
{
    uint8_t buffer[2];

    buffer[0] = n;
    buffer[1] = n >> 8;

    outFile.write((char *) buffer, 2);
}

void AviEncoder::writeChars(const char *s)
{
    size_t count = strlen(s);
    if (count > 255)
    {
        count = 255;
    }
    outFile.write(s, count);
}

void AviEncoder::writeCharsBin(const char *s, int count)
{
    outFile.write(s, count);
}

void VideoEncoder::open(const std::string &videoFile, const uint32_t &width, const uint32_t &height, const VideoType &videoType, const uint8_t &videoJpgQuality, const VideoFpsType &videoFpsType, const VideoMatChannel &videoMatChannel)
{
    if(videoFile == "")
    {
        throw Exception(1, "[Video Encoder]: File name can't be empty! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(width < 1 || height < 1)
    {
        throw Exception(1, "[Video Encoder]: Video width and height must > 0! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(videoJpgQuality>100)
    {
        throw Exception(1, "[Video Encoder]: Video mjpg quality must <= 100! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(this->_inited)
    {
        close();
    }
    this->_width    = width;
    this->_height   = height;
    this->_bpp      = getBpp(videoMatChannel);
    this->_matType  = getMatType(videoMatChannel);
    this->_fps      = getFps(videoFpsType);
    this->_videType = videoType;
    this->_videoJpgQuality = videoJpgQuality;

    aviEncoder = new AviEncoder(videoFile.data(), width, height, this->_bpp, getFourcc(videoType).data(), this->_fps, nullptr);
    this->_inited = true;
}

VideoEncoder::~VideoEncoder()
{
    if(this->_inited)
    {
        close();
    }
}

void bufferFromCallback(void* context, void* data, int size)
{
    unsigned char* ptr = static_cast<unsigned char*>(data);
    std::copy(ptr, ptr + size, std::back_inserter(*static_cast<std::vector<unsigned char>*>(context)));
}

void VideoEncoder::writeMat(const Mat &mat)
{
    Mat tmpMat = mat;
    if(!this->_inited)
    {
        throw Exception(1, "[Video Encoder]: Encoder not opened! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(tmpMat.isEmpty())
    {
        throw Exception(1, "[Video Encoder]: Mat empty! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(tmpMat.getWidth()!=this->_width || tmpMat.getHeight()!=this->_height)
    {
        throw Exception(1, "[Video Encoder]: mat's width or height != enoder's width or height! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(!tmpMat.isU8Mat())
    {
        tmpMat.convertTo(tmpMat, CVT_DATA_TO_U8);
    }

    if(tmpMat.getMatType()!=this->_matType)
    {
        if(this->_bpp == 8)
        {
            if(tmpMat.getMatType() == MAT_RGB_U8)
            {
                MatOp::cvtColor(tmpMat,tmpMat,CVT_RGB2GRAY);
            }
            else if(tmpMat.getMatType() == MAT_RGBA_U8)
            {
                MatOp::cvtColor(tmpMat,tmpMat,CVT_RGBA2GRAY);
            }
        }
        else if(this->_bpp == 24)
        {
            if(tmpMat.getMatType() == MAT_GRAY_U8)
            {
                MatOp::cvtColor(tmpMat,tmpMat,CVT_GRAY2RGB);
            }
            else if(tmpMat.getMatType() == MAT_RGBA_U8)
            {
                MatOp::cvtColor(tmpMat,tmpMat,CVT_RGBA2RGB);
            }
        }
        else if(this->_bpp == 32)
        {
            if(tmpMat.getMatType() == MAT_GRAY_U8)
            {
                MatOp::cvtColor(tmpMat,tmpMat,CVT_GRAY2RGBA);
            }
            else if(tmpMat.getMatType() == MAT_RGB_U8)
            {
                MatOp::cvtColor(tmpMat,tmpMat,CVT_RGB2RGBA);
            }
        }
    }

    std::vector<char> picData;
    if(this->_videType == VIDEO_MJPG)
    {
        stbi_write_jpg_to_func(bufferFromCallback,&picData, this->_width, this->_height, this->_bpp/8 ,tmpMat.getData().u8,this->_videoJpgQuality);
    }
    else if(this->_videType == VIDEO_MPNG)
    {
        stbi_write_png_to_func(bufferFromCallback,&picData, this->_width, this->_height, this->_bpp/8 ,tmpMat.getData().u8,0);
    }

    const char *res = picData.data();
    aviEncoder->addVideoFrame(res,picData.size());
}

void VideoEncoder::close()
{
    aviEncoder->finalize();
    if(aviEncoder!=nullptr)
    {
        delete aviEncoder;
        aviEncoder = nullptr;
    }
    this->_inited  = false;
}

int VideoEncoder::getFps(const VideoFpsType &fpsType)
{
    switch (fpsType)
    {
    case VIDEO_FPS_10:
        return 10;
    case VIDEO_FPS_15:
        return 15;
    case VIDEO_FPS_20:
        return 20;
    case VIDEO_FPS_24:
        return 24;
    case VIDEO_FPS_25:
        return 25;
    case VIDEO_FPS_30:
        return 30;
    case VIDEO_FPS_50:
        return 50;
    case VIDEO_FPS_60:
        return 60;
    default:
        return 24;
    }
}

std::string VideoEncoder::getFourcc(const VideoType &videoType)
{
    switch (videoType)
    {
    case VIDEO_MJPG:
        return "MJPG";
    case VIDEO_MPNG:
        return "MPNG";
    default:
        return "MJPG";
    }
}

int VideoEncoder::getBpp(const VideoMatChannel &aviMatChannel)
{
    switch (aviMatChannel)
    {
    case VIDEO_MAT_GRAY:
        return 8;
    case VIDEO_MAT_RGB:
        return 24;
    case VIDEO_MAT_RGBA:
        return 32;
    default:
        return 24;
    }
}

MatType VideoEncoder::getMatType(const VideoMatChannel &aviMatChannel)
{
    switch (aviMatChannel)
    {
    case VIDEO_MAT_GRAY:
        return MAT_GRAY_U8;
    case VIDEO_MAT_RGB:
        return MAT_RGB_U8;
    case VIDEO_MAT_RGBA:
        return MAT_RGBA_U8;
    default:
        return MAT_RGB_U8;
    }
}

VideoDecoder::~VideoDecoder()
{
    if(_plm!=nullptr)
    {
        plm_destroy(reinterpret_cast<plm_t*>(_plm));
        _plm = nullptr;
    }
}

void VideoDecoder::open(const std::string &mpeg1VideoFile)
{
    this->_opened = false;
    _plm = plm_create_with_filename(mpeg1VideoFile.c_str());

    if (!_plm)
    {
        throw Exception(1,"[Video Decoder]: Open mpeg1 video file error! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    plm_set_audio_enabled(reinterpret_cast<plm_t*>(_plm), 0);

    this->_width  = plm_get_width(reinterpret_cast<plm_t*>(_plm));
    this->_height = plm_get_height(reinterpret_cast<plm_t*>(_plm));

    if(this->_width == 0 || this->_height == 0)
    {
        throw Exception(1,"[Video Decoder]: Video type not supported! Please use FFmpeg to transform like this : ffmpeg -i a.avi -c:v mpeg1video -c:a mp2 -format mpeg -p 2 output.mpg \n",__FILE__,__LINE__,__FUNCTION__);
    }

    this->_opened = true;
}

bool VideoDecoder::getMat(Mat &mat)
{
    if(!this->_opened)
    {
        return false;
    }

    size_t hh = plm_buffer_get_remaining(reinterpret_cast<plm_t*>(_plm)->video_buffer);

    if(hh == 0)
    {
        this->_opened = false;
        return false;
    }
    else
    {
        plm_frame_t *frame = plm_decode_video(reinterpret_cast<plm_t*>(_plm));
        if(frame == nullptr)
        {
            return false;
        }

        mat = Mat(this->_width, this->_height, MAT_RGB_U8);
        plm_frame_to_rgb(frame, mat.getData().u8, this->_width * this->_ch);
        return true;
    }

}

GifEncoder::~GifEncoder()
{
    close();
}

void GifEncoder::open(const std::string &fileName, const uint32_t &width, const uint32_t &height, const bool &useLocalPlatte, const uint16_t &delay, const uint8_t &loop, const uint8_t &palSize)
{
    gif = gifStart(fileName.c_str(), width, height, loop, palSize);
    if(!gif.fp)
    {
        throw Exception(1, "[Gif Encoder]: Open file error : " + fileName + "\n",__FILE__,__LINE__,__FUNCTION__);
    }
    this->_delay = delay;
    this->_width = width;
    this->_height = height;
    this->_uselocalPalette = useLocalPlatte;
}

void GifEncoder::writeMat(const Mat &mat)
{

    if(mat.isEmpty())
    {
        throw Exception(1, "[Gif Encoder]: Mat empty! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    if(static_cast<uint32_t>(mat.getWidth())!=this->_width || static_cast<uint32_t>(mat.getHeight())!=this->_height)
    {
        throw Exception(1, "[Gif Encoder]: mat's width or height != enoder's width or height! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    Mat tmpMat = mat;

    if(!tmpMat.isU8Mat())
    {
        tmpMat.convertTo(tmpMat, CVT_DATA_TO_U8);
    }

    if(tmpMat.getMatType()!=MAT_RGBA_U8)
    {
        if(tmpMat.getMatType() == MAT_GRAY_U8)
        {
            MatOp::cvtColor(tmpMat,tmpMat,CVT_GRAY2RGB);
        }
        else if(tmpMat.getMatType() == MAT_RGB_U8)
        {
            MatOp::cvtColor(tmpMat,tmpMat,CVT_RGB2RGBA);
        }
    }
    gifFrame(&gif, tmpMat.getData().u8, this->_delay, this->_uselocalPalette);
}

void GifEncoder::close()
{
    gifEnd(&gif);
}

void GifEncoder::gifQuantize(uint8_t *rgba, const int &rgbaSize, int sample, uint8_t *map, const int &numColors)
{

    const int intbiasshift = 16; 

    const int intbias = (((int) 1) << intbiasshift);
    const int gammashift = 10; 

    const int betashift = 10;
    const int beta = (intbias >> betashift); 

    const int betagamma = (intbias << (gammashift - betashift));

    const int radiusbiasshift = 6; 

    const int radiusbias = (((int) 1) << radiusbiasshift);
    const int radiusdec = 30; 

    const int alphabiasshift = 10; 

    const int initalpha = (((int) 1) << alphabiasshift);

    const int radbiasshift = 8;
    const int radbias = (((int) 1) << radbiasshift);
    const int alpharadbshift = (alphabiasshift + radbiasshift);
    const int alpharadbias = (((int) 1) << alpharadbshift);

    sample = sample < 1 ? 1 : sample > 30 ? 30 : sample;
    int network[256][3];
    int bias[256] = {}, freq[256];
    for(int i = 0; i < numColors; ++i)
    {

        network[i][0] = network[i][1] = network[i][2] = (i << 12) / numColors;
        freq[i] = intbias / numColors;
    }

    {
        const int primes[5] = {499, 491, 487, 503};
        int step = 4;

        for(int i = 0; i < 4; ++i)
        {
            if(rgbaSize > primes[i] * 4 && (rgbaSize % primes[i]))
            { 

                step = primes[i] * 4;
            }
        }
        sample = step == 4 ? 1 : sample;

        int alphadec = 30 + ((sample - 1) / 3);
        int samplepixels = rgbaSize / (4 * sample);
        int delta = samplepixels / 100;
        int alpha = initalpha;
        delta = delta == 0 ? 1 : delta;

        int radius = (numColors >> 3) * radiusbias;
        int rad = radius >> radiusbiasshift;
        rad = rad <= 1 ? 0 : rad;
        int radSq = rad*rad;
        int radpower[32];

        for (int i = 0; i < rad; i++)
        {
            radpower[i] = alpha * (((radSq - i * i) * radbias) / radSq);
        }

        for(int i = 0, pix = 0; i < samplepixels;)
        {
            int r = rgba[pix + 0] << 4;
            int g = rgba[pix + 1] << 4;
            int b = rgba[pix + 2] << 4;
            int j = -1;
            {

                int bestd = 0x7FFFFFFF, bestbiasd = 0x7FFFFFFF, bestpos = -1;
                for (int k = 0; k < numColors; k++)
                {
                    int *n = network[k];
                    int dist = abs(n[0] - r) + abs(n[1] - g) + abs(n[2] - b);

                    if (dist < bestd)
                    {
                        bestd = dist;
                        bestpos = k;
                    }

                    int biasdist = dist - ((bias[k]) >> (intbiasshift - 4));
                    if (biasdist < bestbiasd)
                    {
                        bestbiasd = biasdist;
                        j = k;
                    }
                    int betafreq = freq[k] >> betashift;
                    freq[k] -= betafreq;
                    bias[k] += betafreq << gammashift;
                }
                freq[bestpos] += beta;
                bias[bestpos] -= betagamma;
            }

            network[j][0] -= (network[j][0] - r) * alpha / initalpha;
            network[j][1] -= (network[j][1] - g) * alpha / initalpha;
            network[j][2] -= (network[j][2] - b) * alpha / initalpha;

            if (rad != 0)
            {

                int lo = j - rad;
                lo = lo < -1 ? -1 : lo;
                int hi = j + rad;
                hi = hi > numColors ? numColors : hi;
                for(int jj = j+1, m=1; jj < hi; ++jj)
                {
                    int a = radpower[m++];
                    network[jj][0] -= (network[jj][0] - r) * a / alpharadbias;
                    network[jj][1] -= (network[jj][1] - g) * a / alpharadbias;
                    network[jj][2] -= (network[jj][2] - b) * a / alpharadbias;
                }

                for(int k = j-1, m=1; k > lo; --k)
                {
                    int a = radpower[m++];
                    network[k][0] -= (network[k][0] - r) * a / alpharadbias;
                    network[k][1] -= (network[k][1] - g) * a / alpharadbias;
                    network[k][2] -= (network[k][2] - b) * a / alpharadbias;
                }
            }

            pix += step;
            pix = pix >= rgbaSize ? pix - rgbaSize : pix;

            if(++i % delta == 0)
            {
                alpha -= alpha / alphadec;
                radius -= radius / radiusdec;
                rad = radius >> radiusbiasshift;
                rad = rad <= 1 ? 0 : rad;
                radSq = rad*rad;

                for (j = 0; j < rad; j++)
                {
                    radpower[j] = alpha * ((radSq - j * j) * radbias / radSq);
                }
            }
        }
    }

    for (int i = 0; i < numColors; i++)
    {
        map[i*3+0] = network[i][0] >>= 4;
        map[i*3+1] = network[i][1] >>= 4;
        map[i*3+2] = network[i][2] >>= 4;
    }
}

void GifEncoder::gifLzwWrite(GifEncoder::GifLzw *s, const int &code)
{
    s->outBits |= code << s->curBits;
    s->curBits += s->numBits;
    while(s->curBits >= 8)
    {
        s->buf[s->idx++] = s->outBits & 255;
        s->outBits >>= 8;
        s->curBits -= 8;

        if (s->idx >= 255)
        {
            putc(s->idx, s->fp);
            fwrite(s->buf, s->idx, 1, s->fp);
            s->idx = 0;
        }
    }
}

void GifEncoder::gifLzwEncode(uint8_t *in, int len, FILE *fp)
{
    GifLzw state;
    state.fp = fp;
    state.numBits = 9;

    int maxcode = 511;

    const int hashSize = 5003;
    uint16_t codetab[hashSize];
    int hashTbl[hashSize];
    memset(hashTbl, 0xFF, sizeof(hashTbl));

    gifLzwWrite(&state, 0x100);

    int free_ent = 0x102;
    int ent = *in++;
CONTINUE:
    while (--len)
    {
        int c = *in++;
        int fcode = (c << 12) + ent;
        int key = (c << 4) ^ ent; 

        while(hashTbl[key] >= 0)
        {
            if(hashTbl[key] == fcode)
            {
                ent = codetab[key];
                goto CONTINUE;
            }
            ++key;
            key = key >= hashSize ? key - hashSize : key;
        }

        gifLzwWrite(&state, ent);
        ent = c;

        if(free_ent < 4096)
        {
            if(free_ent > maxcode)
            {
                ++state.numBits;
                if(state.numBits == 12)
                {
                    maxcode = 4096;
                }
                else
                {
                    maxcode = (1<<state.numBits)-1;
                }
            }
            codetab[key] = free_ent++;
            hashTbl[key] = fcode;
        }
        else
        {
            memset(hashTbl, 0xFF, sizeof(hashTbl));
            free_ent = 0x102;
            gifLzwWrite(&state, 0x100);
            state.numBits = 9;
            maxcode = 511;
        }
    }
    gifLzwWrite(&state, ent);
    gifLzwWrite(&state, 0x101);
    gifLzwWrite(&state, 0);

    if(state.idx)
    {
        putc(state.idx, fp);
        fwrite(state.buf, state.idx, 1, fp);
    }
}

GifEncoder::Gif GifEncoder::gifStart(const char *filename, const uint16_t &width, const uint16_t &height, const uint16_t &repeat, int numColors)
{
    numColors = numColors > 255 ? 255 : numColors < 2 ? 2 : numColors;
    Gif gif = {};
    gif.width = width;
    gif.height = height;
    gif.repeat = repeat;
    gif.numColors = numColors;
    gif.palSize = log2(numColors);

    gif.fp = fopen(filename, "wb");
    if(!gif.fp)
    {
        return gif;
    }

    fwrite("GIF89a", 6, 1, gif.fp);

    fwrite(&gif.width, 2, 1, gif.fp);
    fwrite(&gif.height, 2, 1, gif.fp);
    putc(0xF0 | gif.palSize, gif.fp);
    fwrite("\x00\x00", 2, 1, gif.fp); 

    return gif;
}

void GifEncoder::gifFrame(GifEncoder::Gif *gif, uint8_t *rgba, const uint16_t &delayCsec, const bool &localPalette)
{
    if(!gif->fp)
    {
        return;
    }
    uint16_t width = gif->width;
    uint16_t height = gif->height;
    int size = width * height;

    uint8_t localPalTbl[0x300];
    uint8_t *palette = gif->frame == 0 || !localPalette ? gif->palette : localPalTbl;

    if(gif->frame == 0 || localPalette)
    {
        gifQuantize(rgba, size*4, 1, palette, gif->numColors);
    }

    uint8_t *indexedPixels = (uint8_t *)malloc(size);
    {
        uint8_t *ditheredPixels = (uint8_t*)malloc(size*4);
        memcpy(ditheredPixels, rgba, size*4);

        for(int k = 0; k < size*4; k+=4)
        {
            int rgb[3] = { ditheredPixels[k+0], ditheredPixels[k+1], ditheredPixels[k+2] };
            int bestd = 0x7FFFFFFF, best = -1;

            for(int i = 0; i < gif->numColors; ++i)
            {
                int bb = palette[i*3+0]-rgb[0];
                int gg = palette[i*3+1]-rgb[1];
                int rr = palette[i*3+2]-rgb[2];
                int d = bb*bb + gg*gg + rr*rr;

                if(d < bestd)
                {
                    bestd = d;
                    best = i;
                }
            }
            indexedPixels[k/4] = best;
            int diff[3] = { ditheredPixels[k+0] - palette[indexedPixels[k/4]*3+0], ditheredPixels[k+1] - palette[indexedPixels[k/4]*3+1], ditheredPixels[k+2] - palette[indexedPixels[k/4]*3+2] };

            if(k+4 < size*4)
            {
                ditheredPixels[k+4+0] = (uint8_t)gifClamp(ditheredPixels[k+4+0]+(diff[0]*7/16), 0, 255);
                ditheredPixels[k+4+1] = (uint8_t)gifClamp(ditheredPixels[k+4+1]+(diff[1]*7/16), 0, 255);
                ditheredPixels[k+4+2] = (uint8_t)gifClamp(ditheredPixels[k+4+2]+(diff[2]*7/16), 0, 255);
            }

            if(k+width*4+4 < size*4)
            {
                for(int i = 0; i < 3; ++i)
                {
                    ditheredPixels[k-4+width*4+i] = (uint8_t)gifClamp(ditheredPixels[k-4+width*4+i]+(diff[i]*3/16), 0, 255);
                    ditheredPixels[k+width*4+i] = (uint8_t)gifClamp(ditheredPixels[k+width*4+i]+(diff[i]*5/16), 0, 255);
                    ditheredPixels[k+width*4+4+i] = (uint8_t)gifClamp(ditheredPixels[k+width*4+4+i]+(diff[i]*1/16), 0, 255);
                }
            }
        }
        free(ditheredPixels);
    }
    if(gif->frame == 0)
    {

        fwrite(palette, 3*(1<<(gif->palSize+1)), 1, gif->fp);
        if(gif->repeat >= 0)
        {

            fwrite("\x21\xff\x0bNETSCAPE2.0\x03\x01", 16, 1, gif->fp);
            fwrite(&gif->repeat, 2, 1, gif->fp); 

            putc(0, gif->fp); 

        }
    }

    fwrite("\x21\xf9\x04\x00", 4, 1, gif->fp);
    fwrite(&delayCsec, 2, 1, gif->fp); 

    fwrite("\x00\x00", 2, 1, gif->fp); 

    fwrite("\x2c\x00\x00\x00\x00", 5, 1, gif->fp); 

    fwrite(&width, 2, 1, gif->fp);
    fwrite(&height, 2, 1, gif->fp);
    if (gif->frame == 0 || !localPalette)
    {
        putc(0, gif->fp);
    }
    else
    {
        putc(0x80|gif->palSize, gif->fp );
        fwrite(palette, 3*(1<<(gif->palSize+1)), 1, gif->fp);
    }

    putc(8, gif->fp); 

    gifLzwEncode(indexedPixels, size, gif->fp);
    putc(0, gif->fp); 

    ++gif->frame;
    free(indexedPixels);
}

void GifEncoder::gifEnd(GifEncoder::Gif *gif)
{
    if(!gif->fp)
    {
        return;
    }

    putc(0x3b, gif->fp); 

    fclose(gif->fp);
}

GifDecoder::~GifDecoder()
{
    close();
}

void GifDecoder::open(const std::string &filename)
{
    std::ifstream readFile(filename,std::ios::in);
    if(!readFile.is_open())
    {
        throw Exception(0,"[Gif Decoder] file open filed! \n" + std::string(filename), __FILE__, __LINE__, __FUNCTION__);
    }

    readFile.seekg(0, std::ios::end);
    auto fsize = readFile.tellg();
    readFile.seekg(0, std::ios::beg);

    if (fsize < 1)
    {
        throw Exception(0,"[Gif Decoder] file read filed! \n" + std::string(filename), __FILE__, __LINE__, __FUNCTION__);
    }

    char *data = new char[static_cast<size_t>(fsize)]();
    readFile.read(data, fsize);

    stbi_uc* tmpData = stbi_load_gif_from_memory(reinterpret_cast<stbi_uc*>(data),
                                                 fsize,
                                                 &this->_delay,
                                                 &this->_width,
                                                 &this->_height,
                                                 &this->_frames,
                                                 &this->_ch,
                                                 0);

    delete[] data;
    data = nullptr;

    if(!tmpData)
    {
        throw Exception(0,"[Gif Decoder] file is not a gif! \n"  + std::string(filename) + "\n", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_gifData  = tmpData;
    this->_curFrame = 0;
    this->_allData  = this->_width*this->_height*this->_ch;
}

void GifDecoder::close()
{
    if(this->_gifData!=nullptr)
    {
        delete[] this->_gifData;
        this->_gifData = nullptr;
    }

    if(this->_delay!=nullptr)
    {
        delete this->_delay;
        this->_delay = nullptr;
    }
}

Mat GifDecoder::getMat(const int &index)
{
    if(this->_gifData==nullptr)
    {
        throw Exception(0,"[Gif Decoder] no gif pic read! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(index >= this->_frames)
    {
        throw Exception(0,"[Gif Decoder] index out of frames", __FILE__, __LINE__, __FUNCTION__);
    }

    MatType matType = MatType::MAT_RGB_U8;

    if(this->_ch == 1)
    {
        matType = MatType::MAT_GRAY_U8;
    }
    else if(this->_ch == 4)
    {
        matType = MatType::MAT_RGBA_U8;
    }

    Mat mat(this->_width, this->_height, matType, this->_gifData + this->_allData*index);

    return mat;
}

bool GifDecoder::getNextFrame(Mat &mat)
{
    if(this->_curFrame>=this->_frames)
    {
        return false;
    }

    mat = getMat(this->_curFrame);
    this->_curFrame++;

    return true;
}

#ifdef _WIN32
WinCamera::WinCamera()
{
    _connected = false;
    _width = 0;
    _height = 0;
    _lock = false;
    _changed = false;
    _bufferSize = 0;
    _nullFilter = NULL;
    _mediaEvent = NULL;
    _sampleGrabberFilter = NULL;
    _graph = NULL;

    _isOpen = false;

    CoInitialize(NULL);
}

WinCamera::~WinCamera()
{
    closeCamera();
    CoUninitialize();
}

bool WinCamera::openCamera(const int &camId, const int &width, const int &height, const bool &displayProperties)
{
    HRESULT hr = S_OK;

    CoInitialize(NULL);

    hr = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC,
                          IID_IGraphBuilder, (void **)&_graph);

    hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER,
                          IID_IBaseFilter, (LPVOID *)&_sampleGrabberFilter);

    hr = _graph->QueryInterface(IID_IMediaControl, (void **) &_mediaControl);
    hr = _graph->QueryInterface(IID_IMediaEvent, (void **) &_mediaEvent);

    hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER,
                          IID_IBaseFilter, (LPVOID*) &_nullFilter);

    hr = _graph->AddFilter(_nullFilter, L"NullRenderer");

    hr = _sampleGrabberFilter->QueryInterface(IID_ISampleGrabber, (void**)&_sampleGrabber);

    AM_MEDIA_TYPE   mt;
    ZeroMemory(&mt, sizeof(AM_MEDIA_TYPE));
    mt.majortype = MEDIATYPE_Video;
    mt.subtype = MEDIASUBTYPE_RGB24;
    mt.formattype = FORMAT_VideoInfo;
    hr = _sampleGrabber->SetMediaType(&mt);
    MYFREEMEDIATYPE(mt);

    _graph->AddFilter(_sampleGrabberFilter, L"Grabber");

    bindFilter(camId, &_deviceFilter);
    _graph->AddFilter(_deviceFilter, NULL);

    CComPtr<IEnumPins> pEnum;
    _deviceFilter->EnumPins(&pEnum);

    hr = pEnum->Reset();
    hr = pEnum->Next(1, &_cameraOutput, NULL);

    pEnum = NULL;
    _sampleGrabberFilter->EnumPins(&pEnum);
    pEnum->Reset();
    hr = pEnum->Next(1, &_grabberInput, NULL);

    pEnum = NULL;
    _sampleGrabberFilter->EnumPins(&pEnum);
    pEnum->Reset();
    pEnum->Skip(1);
    hr = pEnum->Next(1, &_grabberOutput, NULL);

    pEnum = NULL;
    _nullFilter->EnumPins(&pEnum);
    pEnum->Reset();
    hr = pEnum->Next(1, &_nullInputPin, NULL);

    if (displayProperties)
    {
        CComPtr<ISpecifyPropertyPages> pPages;

        HRESULT hr = _cameraOutput->QueryInterface(IID_ISpecifyPropertyPages, (void**)&pPages);
        if (SUCCEEDED(hr))
        {
            PIN_INFO PinInfo;
            _cameraOutput->QueryPinInfo(&PinInfo);

            CAUUID caGUID;
            pPages->GetPages(&caGUID);

            OleCreatePropertyFrame(NULL, 0, 0,
                                   L"Property Sheet", 1,
                                   (IUnknown **)&(_cameraOutput.p),
                                   caGUID.cElems,
                                   caGUID.pElems,
                                   0, 0, NULL);
            CoTaskMemFree(caGUID.pElems);
            PinInfo.pFilter->Release();
        }
        pPages = NULL;
    }
    else
    {

        int _Width = width, _Height = height;
        IAMStreamConfig*   iconfig;
        iconfig = NULL;
        hr = _cameraOutput->QueryInterface(IID_IAMStreamConfig,   (void**)&iconfig);

        AM_MEDIA_TYPE* pmt;
        if(iconfig->GetFormat(&pmt) !=S_OK)
        {

            return   false;
        }

        VIDEOINFOHEADER*   phead;
        if ( pmt->formattype == FORMAT_VideoInfo)
        {
            phead=( VIDEOINFOHEADER*)pmt->pbFormat;
            phead->bmiHeader.biWidth = _Width;
            phead->bmiHeader.biHeight = _Height;
            if(( hr=iconfig->SetFormat(pmt)) != S_OK )
            {
                return   false;
            }

        }

        iconfig->Release();
        iconfig=NULL;
        MYFREEMEDIATYPE(*pmt);
    }

    hr = _graph->Connect(_cameraOutput, _grabberInput);
    hr = _graph->Connect(_grabberOutput, _nullInputPin);

    if (FAILED(hr))
    {
        switch(hr)
        {
        case VFW_S_NOPREVIEWPIN :
            break;
        case E_FAIL :
            break;
        case E_INVALIDARG :
            break;
        case E_POINTER :
            break;
        }
    }

    _sampleGrabber->SetBufferSamples(TRUE);
    _sampleGrabber->SetOneShot(TRUE);

    hr = _sampleGrabber->GetConnectedMediaType(&mt);
    if(FAILED(hr))
        return false;

    VIDEOINFOHEADER *videoHeader;
    videoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mt.pbFormat);
    _width = videoHeader->bmiHeader.biWidth;
    _height = videoHeader->bmiHeader.biHeight;
    _connected = true;

    pEnum = NULL;

    _isOpen = true;
    return true;
}

void WinCamera::closeCamera()
{
    if(_connected)
        _mediaControl->Stop();

    _graph = NULL;
    _deviceFilter = NULL;
    _mediaControl = NULL;
    _sampleGrabberFilter = NULL;
    _sampleGrabber = NULL;
    _grabberInput = NULL;
    _grabberOutput = NULL;
    _cameraOutput = NULL;
    _mediaEvent = NULL;
    _nullFilter = NULL;
    _nullInputPin = NULL;

    _connected = false;
    _width = 0;
    _height = 0;
    _lock = false;
    _changed = false;
    _bufferSize = 0;

    _isOpen = false;
}

void WinCamera::getMat(Mat &mat)
{
    if (!_isOpen)
    {
        throw Exception(0,"[Win Camera] cam not opened! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    long evCode;
    long size = 0;

    this->_mediaControl->Run();
    this->_mediaEvent->WaitForCompletion(INFINITE, &evCode);

    this->_sampleGrabber->GetCurrentBuffer(&size, NULL);

    if(size!=_bufferSize)
    {
        if(!mat.isEmpty())
        {
            mat.release();
        }
        _bufferSize=size;
        mat = Mat(_width,_height,MAT_RGB_U8);
    }

    _sampleGrabber->GetCurrentBuffer(&_bufferSize, (long*)mat.getData().u8);
    MatOp::cvtColor(mat,mat,CVT_RGB2BGR);
    MatOp::flip(mat,FLIP_V);
}

int WinCamera::listCameras()
{
    int count = 0;
    CoInitialize(NULL);

    CComPtr<ICreateDevEnum> pCreateDevEnum;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, (void**)&pCreateDevEnum);

    CComPtr<IEnumMoniker> pEm;
    hr = pCreateDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
                                               &pEm, 0);
    if (hr != NOERROR)
    {
        return count;
    }

    pEm->Reset();
    ULONG cFetched;
    IMoniker *pM;
    while(hr = pEm->Next(1, &pM, &cFetched), hr==S_OK)
    {
        count++;
    }

    pCreateDevEnum = NULL;
    pEm = NULL;
    return count;
}

std::string WinCamera::getCameraName(const int &nCamID)
{
    int count = 0;
    CoInitialize(NULL);

    CComPtr<ICreateDevEnum> pCreateDevEnum;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, (void**)&pCreateDevEnum);

    CComPtr<IEnumMoniker> pEm;
    hr = pCreateDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
                                               &pEm, 0);
    if (hr != NOERROR) return 0;

    char* name = new char[1024]();

    pEm->Reset();
    ULONG cFetched;
    IMoniker *pM;
    while(hr = pEm->Next(1, &pM, &cFetched), hr==S_OK)
    {
        if (count == nCamID)
        {
            IPropertyBag *pBag=0;
            hr = pM->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pBag);
            if(SUCCEEDED(hr))
            {
                VARIANT var;
                var.vt = VT_BSTR;
                hr = pBag->Read(L"FriendlyName", &var, NULL); 

                if(hr == NOERROR)
                {

                    WideCharToMultiByte(CP_ACP,0,var.bstrVal,-1,name, 1024 ,"",NULL);

                    SysFreeString(var.bstrVal);
                }
                pBag->Release();
            }
            pM->Release();

            break;
        }
        count++;
    }

    pCreateDevEnum = NULL;
    pEm = NULL;

    std::string nameStr = name;
    delete[] name;
    name = nullptr;
    return nameStr;
}

bool WinCamera::bindFilter(int nCamID, IBaseFilter **pFilter)
{
    if (nCamID < 0)
        return false;

    CComPtr<ICreateDevEnum> pCreateDevEnum;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, (void**)&pCreateDevEnum);
    if (hr != NOERROR)
    {
        return false;
    }

    CComPtr<IEnumMoniker> pEm;
    hr = pCreateDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
                                               &pEm, 0);
    if (hr != NOERROR)
    {
        return false;
    }

    pEm->Reset();
    ULONG cFetched;
    IMoniker *pM;
    int index = 0;
    while(hr = pEm->Next(1, &pM, &cFetched), hr==S_OK, index <= nCamID)
    {
        IPropertyBag *pBag;
        hr = pM->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pBag);
        if(SUCCEEDED(hr))
        {
            VARIANT var;
            var.vt = VT_BSTR;
            hr = pBag->Read(L"FriendlyName", &var, NULL);
            if (hr == NOERROR)
            {
                if (index == nCamID)
                {
                    pM->BindToObject(0, 0, IID_IBaseFilter, (void**)pFilter);
                }
                SysFreeString(var.bstrVal);
            }
            pBag->Release();
        }
        pM->Release();
        index++;
    }

    pCreateDevEnum = NULL;
    return true;
}

void WinCamera::setCrossBar()
{
    int i;
    IAMCrossbar *pXBar1 = NULL;
    ICaptureGraphBuilder2 *pBuilder = NULL;

    HRESULT hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL,
                                  CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2,
                                  (void **)&pBuilder);

    if (SUCCEEDED(hr))
    {
        hr = pBuilder->SetFiltergraph(_graph);
    }

    hr = pBuilder->FindInterface(&LOOK_UPSTREAM_ONLY, NULL,
                                 _deviceFilter,IID_IAMCrossbar, (void**)&pXBar1);

    if (SUCCEEDED(hr))
    {
        long OutputPinCount;
        long InputPinCount;
        long PinIndexRelated;
        long PhysicalType;
        long inPort = 0;
        long outPort = 0;

        pXBar1->get_PinCounts(&OutputPinCount,&InputPinCount);
        for( i =0;i<InputPinCount;i++)
        {
            pXBar1->get_CrossbarPinInfo(TRUE,i,&PinIndexRelated,&PhysicalType);
            if(PhysConn_Video_Composite==PhysicalType)
            {
                inPort = i;
                break;
            }
        }
        for( i =0;i<OutputPinCount;i++)
        {
            pXBar1->get_CrossbarPinInfo(FALSE,i,&PinIndexRelated,&PhysicalType);
            if(PhysConn_Video_VideoDecoder==PhysicalType)
            {
                outPort = i;
                break;
            }
        }

        if(S_OK==pXBar1->CanRoute(outPort,inPort))
        {
            pXBar1->Route(outPort,inPort);
        }
        pXBar1->Release();
    }
    pBuilder->Release();
}
#endif

#ifdef linux

std::vector<std::string> LinuxCamera::cams;
LinuxCamera::LinuxCamera()
{

}

LinuxCamera::~LinuxCamera()
{
    closeCam();
}

std::string LinuxCamera::system(const std::string &cmd)
{
    char result[10240] = {0};
    char buf[1024] = {0};
    FILE *fp = NULL;

    if( (fp = popen(cmd.data(), "r")) == NULL )
    {
        throw Exception(0,"[Linux Cam] popen error! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    while (fgets(buf, sizeof(buf), fp))
    {
        strcat(result, buf);
    }

    pclose(fp);
    return std::string(result);
}

int LinuxCamera::listCameras()
{
    std::string info = system("ls /dev/ | grep video");
    cams.clear();
    split(cams,info,"\n");
    return cams.size();
}

std::string LinuxCamera::getCameraName(const int &camId)
{
    int fd = 0;
    struct v4l2_capability        cap;               

    int n  = LinuxCamera::listCameras();
    if(camId>n-1)
    {
        throw Exception(0,"[Linux Cam] no cam! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(camId>n-1)
    {
        throw Exception(0,"[Linux Cam] given cam id > cam nums! \n", __FILE__, __LINE__, __FUNCTION__);
    }
    std::string camPath = "/dev/" + LinuxCamera::cams[camId];

    if((fd=open(camPath.data(), O_RDWR)) == -1) 

    {
        throw Exception(0,"[Linux Cam] Error opening V4L interface! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1)

    {
        throw Exception(0,"[Linux Cam] Error opening device "+camPath+": unable to query device.\n", __FILE__, __LINE__, __FUNCTION__);
    }
    if(fd!=-1)
    {
        close(fd);
    }

    return std::string(reinterpret_cast<char*>(cap.card));
}

void LinuxCamera::split(std::vector<std::string> &result, const std::string &str, const std::string &delimiter)
{
    char* save = nullptr;
    char* token = strtok_r(const_cast<char*>(str.c_str()), delimiter.c_str(), &save);
    while (token != nullptr)
    {
        result.emplace_back(token);
        token = strtok_r(nullptr, delimiter.c_str(), &save);
    }
}

bool LinuxCamera::openCamera(const int &camId, const int &width, const int &height, const int &fps)
{
    int n = LinuxCamera::listCameras();
    if(camId>n-1)
    {
        std::cout<<"[Linux Cam] no cam! \n";
        return false;
    }

    if(camId>n-1)
    {
        std::cout<<"[Linux Cam] given cam id > cam nums! \n";
        return false;
    }
    std::string camPath = "/dev/" + LinuxCamera::cams[camId];

    if((fd=open(camPath.data(), O_RDWR)) == -1) 

    {
        std::cout<<"[Linux Cam] Error opening V4L interface! \n";
        return false;
    }

    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1)

    {
        std::cout<<"[Linux Cam] Error opening device "<<camPath<<": unable to query device.\n";
        return false;
    }

    fmtdesc.index=0;
    fmtdesc.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;          

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;            

    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;       

    fmt.fmt.pix.height = height;                   

    fmt.fmt.pix.width = width;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;         

    if(ioctl(fd, VIDIOC_S_FMT, &fmt) == -1)            

    {
        std::cout<<"[Linux Cam] Unable to set format\n";
        return false;
    }

    if(ioctl(fd, VIDIOC_G_FMT, &fmt) == -1)

    {
        std::cout<<"[Linux Cam] Unable to get format\n";
        return false;
    }

    setfps.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    setfps.parm.capture.timeperframe.denominator = fps;  

    setfps.parm.capture.timeperframe.numerator = 1;     

    if(ioctl(fd, VIDIOC_S_PARM, &setfps)==-1)           

    {
        std::cout<<"[Linux Cam] Unable to set fps\n";
        return false;
    }

    req.count   =VIDEO_COUNT;                              

    req.type    =V4L2_BUF_TYPE_VIDEO_CAPTURE;               

    req.memory  =V4L2_MEMORY_MMAP;                        

    if(ioctl(fd,VIDIOC_REQBUFS,&req)==-1)               

    {
        std::cout<<"[Linux Cam] request for buffers error\n";
        return false;
    }

    for (i=0; i<VIDEO_COUNT; i++)       

    {
        bzero(&buffer[i], sizeof(buffer[i]));
        buffer[i].type      = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer[i].memory    = V4L2_MEMORY_MMAP;
        buffer[i].index     = i;
        if (ioctl (fd, VIDIOC_QUERYBUF, &buffer[i]) == -1) 

        {
            std::cout<<"[Linux Cam] query buffer error\n";
            return false;
        }

        length[i]   = buffer[i].length;                   

        start[i]    = (unsigned char *)mmap(NULL,buffer[i].length,PROT_READ |PROT_WRITE, MAP_SHARED, fd, buffer[i].m.offset);

    }

    for (i=0; i<VIDEO_COUNT; i++)
    {
        buffer[i].index = i;
        ioctl(fd, VIDIOC_QBUF, &buffer[i]);             

    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl (fd, VIDIOC_STREAMON, &type);                 

    bzero(&v4lbuf, sizeof(v4lbuf));
    v4lbuf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4lbuf.memory   = V4L2_MEMORY_MMAP;
    this->_height   = height;
    this->_width    = width;
    return true;
}

void LinuxCamera::getMat(Mat &mat)
{
    v4lbuf.index = n%VIDEO_COUNT;
    ioctl(fd, VIDIOC_DQBUF, &v4lbuf);                  

    int matW = 0;
    int matH = 0;
    int matC = 0;

    uint8_t* data = stbi_load_from_memory(reinterpret_cast<stbi_uc*>(start[n%VIDEO_COUNT]),length[n%VIDEO_COUNT],&matW,&matH,&matC,0);

    if(mat.getWidth()!=matW || mat.getHeight()!=matH || mat.getChannel()!=matC)
    {
        if(mat.isEmpty())
        {
            mat.release();
        }

        MatType type;

        if(matC == 1)
        {
            type = MAT_GRAY_U8;
        }
        else if(matC == 3)
        {
            type = MAT_RGB_U8;
        }
        else if(matC == 4)
        {
            type = MAT_RGBA_U8;
        }

        mat = Mat(matW,matH,type);
    }

    if(mat.getBytes()!=nullptr)
    {
        memcpy(mat.getBytes(), data, matW*matH*matC);
    }

    stbi_image_free(data);

    v4lbuf.index = n%VIDEO_COUNT;
    ioctl(fd, VIDIOC_QBUF, &v4lbuf);                   

    n++;
    if(n == 3)
    {                                                   

        n = 0;
    }
}

void LinuxCamera::closeCam()
{
    if(fd != -1)
    {
        ioctl(fd, VIDIOC_STREAMOFF, &type);            

        int n = close(fd);                             

        if(n == -1)
        {
            return ;
        }
    }
    for(i=0; i<VIDEO_COUNT; i++)
    {
        if(start[i] != NULL){                          

            start[i] = NULL;
        }
    }
}

#endif

int VideoCapture::listCameras()
{
#ifdef linux
    return LinuxCamera::listCameras();
#endif

#ifdef _WIN32
    return WinCamera::listCameras();
#endif
}

std::string VideoCapture::getCameraName(const int &camId)
{
#ifdef linux
    return LinuxCamera::getCameraName(camId);
#endif

#ifdef _WIN32
    return WinCamera::getCameraName(camId);
#endif
}

bool VideoCapture::openCamera(const int &camId, const int &width, const int &height, const int &fps)
{
    this->_isOpened = cam.openCamera(camId,width,height,fps);
    return this->_isOpened;
}

void VideoCapture::getMat(Mat &mat)
{
    cam.getMat(mat);
}

}
