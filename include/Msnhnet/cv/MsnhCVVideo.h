#ifndef MSNHCVVIDEO_H
#define MSNHCVVIDEO_H
#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVMatOp.h"
#include "Msnhnet/cv/MsnhCVGui.h"
#include <fstream>
#include <algorithm>
#include <stdio.h>

#ifdef WIN32
#include <windows.h>
#include <ole2.h>
#include <strmif.h>
#include <atlbase.h>
#include <dshow.h>
#include <winnt.h>
#include <control.h>
#else
#include <fcntl.h>                                  

#include <sys/mman.h>                               

#include <linux/videodev2.h>                        

#include <unistd.h>                                 

#include <sys/ioctl.h>
#endif

namespace Msnhnet
{
class MsnhNet_API AviEncoder
{
    struct AviHeader {
        uint32_t timeDelay; 

        uint32_t dataRate; 

        uint32_t reserved;
        uint32_t flags; 

        uint32_t numberOfFrames; 

        uint32_t initialFrames; 

        uint32_t dataStreams; 

        uint32_t bufferSize; 

        uint32_t width; 

        uint32_t height; 

        uint32_t timeScale;
        uint32_t playbackDataRate;
        uint32_t startingTime;
        uint32_t dataLength;
    };
    struct AviStreamHeader {
        char data_type[5]; 

        char codec[5]; 

        uint32_t flags; 

        uint32_t priority;
        uint32_t initialFrames;

        uint32_t timeScale; 

        uint32_t dataRate; 

        uint32_t startTime; 

        uint32_t dataLength; 

        uint32_t bufferSize; 

        uint32_t videoQuality; 

        int audioQuality;
        uint32_t sampleSize; 

    };
    struct AviStreamFormatV {
        uint32_t headerSize;
        uint32_t width;
        uint32_t height;
        uint16_t numPlanes;
        uint16_t bitsPerPixel;
        uint32_t compressionType;
        uint32_t imageSize;
        uint32_t xPelsPerMeter;
        uint32_t yPelsPerMeter;
        uint32_t colorsUsed;
        uint32_t colorsImportant;
        uint32_t *palette;
        uint32_t paletteCount;
    };
    struct AviStreamFormatA {
        uint16_t formatType;
        uint32_t channels;
        uint32_t sampleRate;
        uint32_t bytesPerSecond;
        uint32_t blockAlign;
        uint32_t bitsPerSample;
        uint16_t size;
    };

    typedef struct
    {
        uint32_t channels;
        uint32_t bits;
        uint32_t samplesPerSecond;
    } AviAudio;

public:
    AviEncoder(const char *filename, const uint32_t &width, const uint32_t &height, const uint32_t &bpp, const char *fourcc, const uint32_t &fps, AviAudio *audio);
    ~AviEncoder();

    void addVideoFrame(const char *buffer, const size_t &len);
    void addAudioFrame(const char *buffer, const size_t &len);
    void finalize();
    void setFramerate(const uint32_t &fps);
    void setFourccCodec(const char *fourcc);
    void setVideoFrameSize(const uint32_t &width, const uint32_t &height);

private:
    std::ofstream outFile;
    struct AviHeader aviHeader;
    struct AviStreamHeader streamHeaderV;
    struct AviStreamFormatV streamFormatV;
    struct AviStreamHeader streamHeaderA;
    struct AviStreamFormatA streamFormatA;
    uint32_t marker = 0;
    int offsetsPtr = 0;
    int offsetsLen = 0;
    uint32_t offsetsStart = 0;
    uint32_t *offsets = nullptr;
    int offsetCount = 0;

    void writeAviHeader(struct AviHeader *aviHeader);
    void writeStreamHeader(struct AviStreamHeader *streamHeader);
    void writeStreamFormatV(struct AviStreamFormatV *streamFormatV);
    void writeStreamFormatA(struct AviStreamFormatA *streamFormatA);
    void writeAviHeaderChunk();
    void writeIndex(const uint32_t &count, uint32_t *offsets);
    int checkFourcc(const char *fourcc);

    void writeInt(uint32_t n);
    void writeuint16_t(uint16_t n);
    void writeChars(const char *s);
    void writeCharsBin(const char *s, int count);
};

class MsnhNet_API VideoEncoder
{
public:
    VideoEncoder(){}

    void open(const std::string &videoFile, const uint32_t &width, const uint32_t &height,
              const VideoType &videoType=VIDEO_MJPG,
              const uint8_t &videoJpgQuality=100,
              const VideoFpsType &videoFpsType=VIDEO_FPS_24,
              const VideoMatChannel &videoMatChannel=VIDEO_MAT_RGB
            );

    ~VideoEncoder();

    void writeMat(const Mat &mat);

    void close();

private:
    AviEncoder* aviEncoder = nullptr;
    uint32_t _width  = 0;
    uint32_t _height = 0;
    uint32_t _fps    = 24;
    uint32_t _bpp    = 24;
    bool _inited     = false;
    uint8_t _videoJpgQuality = 100;

    VideoType _videType = VIDEO_MJPG;
    MatType _matType    = MAT_RGB_U8;

    int getFps(const VideoFpsType &fpsType);
    std::string getFourcc(const VideoType &videoType);
    int getBpp(const VideoMatChannel &aviMatChannel);
    MatType getMatType(const VideoMatChannel &aviMatChannel);
};

class MsnhNet_API VideoDecoder
{

public:
    VideoDecoder(){}
    ~VideoDecoder();
    void open(const std::string &mpeg1VideoFile);
    bool getMat(Mat &mat);
private:
    void *_plm = nullptr;
    bool _opened = false;
    uint32_t _width  = 0;
    uint32_t _height = 0;
    const uint8_t _ch = 3;
};

extern void bufferFromCallback(void* context, void* data, int size);

class MsnhNet_API GifEncoder
{
public:
    GifEncoder(){}
    ~GifEncoder();

    void open(const std::string &fileName, const uint32_t &width, const uint32_t &height, const bool &useLocalPlatte=true,
              const uint16_t& delay=4, const uint8_t &loop=0, const uint8_t &palSize=32);
    void writeMat(const Mat& mat);
    void close();
private:
    typedef struct
    {
        FILE *fp;
        uint8_t palette[0x300];
        uint16_t width  = 0;
        uint16_t height = 0;
        uint16_t repeat = 0;
        int numColors   = 0;
        int palSize     = 0;
        int frame       = 0;
    } Gif;

    typedef struct
    {
        FILE *fp;
        int numBits;
        uint8_t buf[256];
        uint8_t idx  = 0;
        uint32_t tmp = 0;
        int outBits  = 0;
        int curBits  = 0;
    } GifLzw;

    void gifQuantize(uint8_t *rgba, const int &rgbaSize, int sample, uint8_t *map, const int &numColors);
    void gifLzwWrite(GifLzw *s, const int &code);
    void gifLzwEncode(uint8_t *in, int len, FILE *fp);
    void gifFrame(Gif *gif, uint8_t *rgba, const uint16_t &delayCsec, const bool &localPalette);
    void gifEnd(Gif *gif);
    int gifClamp(const int &a, const int &b, const int &c) { return a < b ? b : a > c ? c : a; }
    Gif gifStart(const char *filename, const uint16_t &width, const uint16_t &height, const uint16_t &repeat, int numColors);

    Gif gif;

    uint16_t _delay  = 4;
    uint32_t _width  = 0;
    uint32_t _height = 0;
    bool _uselocalPalette = false;

};

class MsnhNet_API GifDecoder
{
public:
    GifDecoder(){}
    ~GifDecoder();
    void open(const std::string &filename);

    void close();

    Mat getMat(const int &index);

    bool getNextFrame(Mat &mat);

private:
    uint8_t *_gifData  = nullptr;
    int  *_delay    = nullptr;
    int  _width     = 0;
    int  _height    = 0;
    int  _frames    = 0;
    int  _ch        = 0;
    int  _curFrame  = 0;
    int  _allData   = 0;
};

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN

interface
ISampleGrabberCB
:
  public IUnknown
{
             virtual STDMETHODIMP SampleCB(double SampleTime, IMediaSample *pSample) = 0;
virtual STDMETHODIMP BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen) = 0;
};

static
const
IID IID_ISampleGrabberCB = { 0x0579154A, 0x2B53, 0x4994,{ 0xB0, 0xD0, 0xE7, 0x73, 0x14, 0x8E, 0xFF, 0x85 } };

interface
ISampleGrabber
:
  public IUnknown
{
             virtual HRESULT STDMETHODCALLTYPE SetOneShot(BOOL OneShot) = 0;
virtual HRESULT STDMETHODCALLTYPE SetMediaType(const AM_MEDIA_TYPE *pType) = 0;
virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(AM_MEDIA_TYPE *pType) = 0;
virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(BOOL BufferThem) = 0;
virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(long *pBufferSize, long *pBuffer) = 0;
virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(IMediaSample **ppSample) = 0;
virtual HRESULT STDMETHODCALLTYPE SetCallback(ISampleGrabberCB *pCallback, long WhichMethodToCallback) = 0;
};

static
const
IID IID_ISampleGrabber = { 0x6B652FFF, 0x11FE, 0x4fce,{ 0x92, 0xAD, 0x02, 0x66, 0xB5, 0xD7, 0xC7, 0x8F } };

static
const
CLSID CLSID_SampleGrabber = { 0xC1F400A0, 0x3F08, 0x11d3,{ 0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37 } };

static
const
CLSID CLSID_NullRenderer = { 0xC1F400A4, 0x3F08, 0x11d3,{ 0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37 } };

static
const
CLSID CLSID_VideoEffects1Category = { 0xcc7bfb42, 0xf175, 0x11d1,{ 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };

static
const
CLSID CLSID_VideoEffects2Category = { 0xcc7bfb43, 0xf175, 0x11d1,{ 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };

static
const
CLSID CLSID_AudioEffects1Category = { 0xcc7bfb44, 0xf175, 0x11d1,{ 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };

static
const
CLSID CLSID_AudioEffects2Category = { 0xcc7bfb45, 0xf175, 0x11d1,{ 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };

#define MYFREEMEDIATYPE(mt)	{if ((mt).cbFormat != 0)		\
{CoTaskMemFree((PVOID)(mt).pbFormat);	\
    (mt).cbFormat = 0;						\
    (mt).pbFormat = NULL;					\
    }											\
    if ((mt).pUnk != NULL)						\
{											\
    (mt).pUnk->Release();					\
    (mt).pUnk = NULL;						\
    }}

class MsnhNet_API WinCamera
{
public:
    WinCamera();
    ~WinCamera();

    static int listCameras();

    static std::string getCameraName(const int &nCamID);

    bool openCamera(const int &camId, const int &width=640, const int &height=480, const bool &displayProperties=false);

    void closeCamera();

    void getMat(Mat &mat);

private:
    bool _isOpen    = false;
    bool _connected = false;
    bool _lock      = false;
    bool _changed   = false;

    int _width      = 0;
    int _height     = 0;
    long _bufferSize= 0;

    Mat _mat;

    CComPtr<IGraphBuilder> _graph;
    CComPtr<IBaseFilter> _deviceFilter;
    CComPtr<IMediaControl> _mediaControl;
    CComPtr<IBaseFilter> _sampleGrabberFilter;
    CComPtr<ISampleGrabber> _sampleGrabber;
    CComPtr<IPin> _grabberInput;
    CComPtr<IPin> _grabberOutput;
    CComPtr<IPin> _cameraOutput;
    CComPtr<IMediaEvent> _mediaEvent;
    CComPtr<IBaseFilter> _nullFilter;
    CComPtr<IPin> _nullInputPin;

    bool bindFilter(int nCamID, IBaseFilter **pFilter);
    void setCrossBar();
};

#endif

#ifdef linux
#define VIDEO_COUNT 3
class MsnhNet_API LinuxCamera
{
public:
    LinuxCamera();
    ~LinuxCamera();

    static std::string system(const std::string& cmd);
    static int listCameras();
    static std::string getCameraName(const int &camId);
    static void split(std::vector<std::string> &result, const std::string& str, const std::string& delimiter);
    bool openCamera(const int &camId, const int &width=640, const int &height=480, const int &fps=30);
    void getMat(Mat &mat);
    void closeCam();
private:
    static std::vector<std::string> cams;

    int                           i         = 0;
    int                           fd        = 0;               

    int                           length[VIDEO_COUNT];

    unsigned char *               start[VIDEO_COUNT];

    struct v4l2_buffer            buffer[VIDEO_COUNT];

    struct v4l2_format            fmt;               

    struct v4l2_fmtdesc           fmtdesc;           

    struct v4l2_capability        cap;               

    struct v4l2_streamparm        setfps;            

    struct v4l2_requestbuffers    req;               

    struct v4l2_buffer            v4lbuf;            

    enum   v4l2_buf_type          type;              

    int                           n         = 0;
    int                           _width    = 0;
    int                           _height   = 0;
};

#endif

class MsnhNet_API VideoCapture
{
public:
    static int listCameras();
    static std::string getCameraName(const int &camId);
    bool openCamera(const int &camId, const int &width=640, const int &height=480, const int &fps=30);
    void getMat(Mat &mat);
    bool isOpened();
private:
#ifdef linux
    LinuxCamera cam;
#endif

#ifdef _WIN32
    WinCamera cam;
#endif

    bool _isOpened = false;
};

}
#endif 

