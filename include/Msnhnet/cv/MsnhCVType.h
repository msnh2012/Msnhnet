#ifndef MSNHCVTYPE_H
#define MSNHCVTYPE_H

#include <stdint.h>

namespace Msnhnet
{
enum MatType
{
    MAT_GRAY_U8,
    MAT_GRAY_F32,
    MAT_RGB_U8,
    MAT_RGB_F32,
    MAT_RGBA_U8,
    MAT_RGBA_F32,
};

enum SaveImageType
{
    MAT_SAVE_PNG,
    MAT_SAVE_BMP,
    MAT_SAVE_JPG,
    MAT_SAVE_HDR,
    MAT_SAVE_TGA,
};

enum CvtColorType
{
    CVT_RGB2GRAY,
    CVT_RGBA2GRAY,
    CVT_GRAY2RGB,
    CVT_GRAY2RGBA,
    CVT_RGBA2RGB,
    CVT_RGB2RGBA,
    CVT_RGB2BGR,
};

enum ResizeType
{
    RESIZE_NEAREST,
    RESIZE_BILINEAR
};

union MatData
{
    uint8_t     *u8 = nullptr;
    int8_t      *i8 ;
    uint16_t    *u16;
    int16_t     *i16;
    uint32_t    *u32;
    int32_t     *i32;
    float       *f32;
};

struct Vec2U8
{
    Vec2U8(){}
    Vec2U8(const uint8_t &x1, const uint8_t &x2):x1(x1),x2(x2){}
    uint8_t x1  = 0;
    uint8_t x2  = 0;
};

struct Vec3U8
{
    Vec3U8(){}
    Vec3U8(const uint8_t &x1, const uint8_t &x2, const uint8_t &x3):x1(x1),x2(x2),x3(x3) {}
    uint8_t x1  = 0;
    uint8_t x2  = 0;
    uint8_t x3  = 0;
};

struct Vec4U8
{
    Vec4U8(){}
    Vec4U8(const uint8_t &x1, const uint8_t &x2, const uint8_t &x3, const uint8_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    uint8_t x1   = 0;
    uint8_t x2   = 0;
    uint8_t x3   = 0;
    uint8_t x4   = 0;
};

struct Vec2U16
{
    Vec2U16(){}
    Vec2U16(const uint16_t &x1, const uint16_t &x2):x1(x1),x2(x2){}
    uint16_t x1  = 0;
    uint16_t x2  = 0;
};

struct Vec3U16
{
    Vec3U16(){}
    Vec3U16(const uint16_t &x1, const uint16_t &x2, const uint16_t &x3):x1(x1),x2(x2),x3(x3) {}
    uint16_t x1  = 0;
    uint16_t x2  = 0;
    uint16_t x3  = 0;
};

struct Vec4U16
{
    Vec4U16(){}
    Vec4U16(const uint16_t &x1, const uint16_t &x2, const uint16_t &x3, const uint16_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    uint16_t x1   = 0;
    uint16_t x2   = 0;
    uint16_t x3   = 0;
    uint16_t x4   = 0;
};

struct Vec2U32
{
    Vec2U32(){}
    Vec2U32(const uint32_t &x1, const uint32_t &x2):x1(x1),x2(x2){}
    uint32_t x1  = 0;
    uint32_t x2  = 0;
};

struct Vec3U32
{
    Vec3U32(){}
    Vec3U32(const uint32_t &x1, const uint32_t &x2, const uint32_t &x3):x1(x1),x2(x2),x3(x3) {}
    uint32_t x1  = 0;
    uint32_t x2  = 0;
    uint32_t x3  = 0;
};

struct Vec4U32
{
    Vec4U32(){}
    Vec4U32(const uint32_t &x1, const uint32_t &x2, const uint32_t &x3, const uint32_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    uint32_t x1   = 0;
    uint32_t x2   = 0;
    uint32_t x3   = 0;
    uint32_t x4   = 0;
};

struct Vec2U64
{
    Vec2U64(){}
    Vec2U64(const uint64_t &x1, const uint64_t &x2):x1(x1),x2(x2){}
    uint64_t x1  = 0;
    uint64_t x2  = 0;
};

struct Vec3U64
{
    Vec3U64(){}
    Vec3U64(const uint64_t &x1, const uint64_t &x2, const uint64_t &x3):x1(x1),x2(x2),x3(x3) {}
    uint64_t x1  = 0;
    uint64_t x2  = 0;
    uint64_t x3  = 0;
};

struct Vec4U64
{
    Vec4U64(){}
    Vec4U64(const uint64_t &x1, const uint64_t &x2, const uint64_t &x3, const uint64_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    uint64_t x1   = 0;
    uint64_t x2   = 0;
    uint64_t x3   = 0;
    uint64_t x4   = 0;
};

struct Vec2I8
{
    Vec2I8(){}
    Vec2I8(const int8_t &x1, const int8_t &x2):x1(x1),x2(x2){}
    int8_t x1  = 0;
    int8_t x2  = 0;
};

struct Vec3I8
{
    Vec3I8(){}
    Vec3I8(const int8_t &x1, const int8_t &x2, const int8_t &x3):x1(x1),x2(x2),x3(x3) {}
    int8_t x1  = 0;
    int8_t x2  = 0;
    int8_t x3  = 0;
};

struct Vec4I8
{
    Vec4I8(){}
    Vec4I8(const int8_t &x1, const int8_t &x2, const int8_t &x3, const int8_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    int8_t x1   = 0;
    int8_t x2   = 0;
    int8_t x3   = 0;
    int8_t x4   = 0;
};

struct Vec2I16
{
    Vec2I16(){}
    Vec2I16(const int16_t &x1, const int16_t &x2):x1(x1),x2(x2){}
    int16_t x1  = 0;
    int16_t x2  = 0;
};

struct Vec3I16
{
    Vec3I16(){}
    Vec3I16(const int16_t &x1, const int16_t &x2, const int16_t &x3):x1(x1),x2(x2),x3(x3) {}
    int16_t x1  = 0;
    int16_t x2  = 0;
    int16_t x3  = 0;
};

struct Vec4I16
{
    Vec4I16(){}
    Vec4I16(const int16_t &x1, const int16_t &x2, const int16_t &x3, const int16_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    int16_t x1   = 0;
    int16_t x2   = 0;
    int16_t x3   = 0;
    int16_t x4   = 0;
};

struct Vec2I32
{
    Vec2I32(){}
    Vec2I32(const int32_t &x1, const int32_t &x2):x1(x1),x2(x2){}
    int32_t x1  = 0;
    int32_t x2  = 0;
};

struct Vec3I32
{
    Vec3I32(){}
    Vec3I32(const int32_t &x1, const int32_t &x2, const int32_t &x3):x1(x1),x2(x2),x3(x3) {}
    int32_t x1  = 0;
    int32_t x2  = 0;
    int32_t x3  = 0;
};

struct Vec4I32
{
    Vec4I32(){}
    Vec4I32(const int32_t &x1, const int32_t &x2, const int32_t &x3, const int32_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    int32_t x1   = 0;
    int32_t x2   = 0;
    int32_t x3   = 0;
    int32_t x4   = 0;
};

struct Vec2I64
{
    Vec2I64(){}
    Vec2I64(const int64_t &x1, const int64_t &x2):x1(x1),x2(x2){}
    int64_t x1  = 0;
    int64_t x2  = 0;
};

struct Vec3I64
{
    Vec3I64(){}
    Vec3I64(const int64_t &x1, const int64_t &x2, const int64_t &x3):x1(x1),x2(x2),x3(x3) {}
    int64_t x1  = 0;
    int64_t x2  = 0;
    int64_t x3  = 0;
};

struct Vec4I64
{
    Vec4I64(){}
    Vec4I64(const int64_t &x1, const int64_t &x2, const int64_t &x3, const int64_t &x4):x1(x1),x2(x2),x3(x3),x4(x4) {}
    int64_t x1   = 0;
    int64_t x2   = 0;
    int64_t x3   = 0;
    int64_t x4   = 0;
};

struct Vec2F32
{
    Vec2F32(){}
    Vec2F32(const float &x1, const float &x2):x1(x1),x2(x2) {}
    float x1  = 0;
    float x2  = 0;
};

struct Vec3F32
{
    Vec3F32(){}
    Vec3F32(const float &x1, const float &x2, const float &x3):x1(x1),x2(x2),x3(x3) {}
    float x1  = 0;
    float x2  = 0;
    float x3  = 0;
};

struct Vec4F32
{
    Vec4F32(){}
    Vec4F32(const float &x1, const float &x2, const float &x3):x1(x1),x2(x2),x3(x3),x4(x4) {}
    float x1   = 0;
    float x2   = 0;
    float x3   = 0;
    float x4   = 0;
};

struct Vec2F64
{
    Vec2F64(){}
    Vec2F64(const double &x1, const double &x2):x1(x1),x2(x2) {}
    double x1  = 0;
    double x2  = 0;
};

struct Vec3F64
{
    Vec3F64(){}
    Vec3F64(const double &x1, const double &x2, const double &x3):x1(x1),x2(x2),x3(x3) {}
    double x1  = 0;
    double x2  = 0;
    double x3  = 0;
};

struct Vec4F64
{
    Vec4F64(){}
    Vec4F64(const double &x1, const double &x2, const double &x3):x1(x1),x2(x2),x3(x3),x4(x4) {}
    double x1   = 0;
    double x2   = 0;
    double x3   = 0;
    double x4   = 0;
};

template<typename T> class DataType
{
public:

};

template<> class DataType<uint8_t>
{
public:
    enum {
           fmt      = (int)'b',
           array    = 1,
           step     = sizeof(uint8_t)
         };
};

template<> class DataType<uint16_t>
{
public:
    enum {
           fmt      = (int)'s',
           array    = 1,
           step     = sizeof(uint16_t)
         };
};

template<> class DataType<uint32_t>
{
public:
    enum {
           fmt      = (int)'l',
           array    = 1,
           step     = sizeof(uint32_t)
         };
};

template<> class DataType<int8_t>
{
public:
    enum {
           fmt      = (int)'B',
           array    = 1,
           step     = sizeof(int8_t)
         };
};

template<> class DataType<int16_t>
{
public:
    enum {
           fmt      = (int)'S',
           array    = 1,
           step     = sizeof(int16_t)
         };
};

template<> class DataType<int32_t>
{
public:
    enum {
           fmt      = (int)'L',
           array    = 1,
           step     = sizeof(int32_t)
         };
};

template<> class DataType<float>
{
public:
    enum {
           fmt      = (int)'f',
           array    = 1,
           step     = sizeof(float)
         };
};

template<> class DataType<Vec3U8>
{
public:
    enum {
           fmt      = (int)'b',
           array    = 3,
           step     = sizeof(uint8_t)*3
         };
};

template<> class DataType<Vec3F32>
{
public:
    enum {
           fmt      = (int)'f',
           array    = 3,
           step     = sizeof(float)*3
         };
};

template<> class DataType<Vec4U8>
{
public:
    enum {
           fmt      = (int)'b',
           array    = 4,
           step     = sizeof(uint8_t)*4
         };
};

template<> class DataType<Vec4F32>
{
public:
    enum {
           fmt      = (int)'f',
           array    = 4,
           step     = sizeof(float)*4
         };
};

}

#endif 

