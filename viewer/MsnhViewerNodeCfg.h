#ifndef MSNHVIEWERNODECFG_H
#define MSNHVIEWERNODECFG_H
#include <MsnhViewerView.h>
#include <MsnhViewerColorTabel.h>
#include <MsnhViewerNodeCreator.h>
namespace MsnhViewer
{
class NodeCfg
{
public:
    static void configNodes()
    {
        NodeCreator::instance().addItem("Inputs",
        {
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Inputs",QColor(0xce4a50));

        NodeCreator::instance().addItem("AddOutputs",
        {
            {"input", "string", AttributeInfo::Type::input},
        });
        ColorTabel::instance().addColor("AddOutputs",QColor(0xB5B83E));

        NodeCreator::instance().addItem("ConcatOutputs",
        {
            {"input", "string", AttributeInfo::Type::input},
        });
        ColorTabel::instance().addColor("ConcatOutputs",QColor(0xCF7531));


        NodeCreator::instance().addItem("Empty",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Empty",QColor(0xB85646));

        NodeCreator::instance().addItem("Activate",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Activate",QColor(0xcc2121));

        NodeCreator::instance().addItem("AddBlock",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("AddBlock",QColor(0x377375));

        NodeCreator::instance().addItem("BatchNorm",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("BatchNorm",QColor(0x5B9FFF));

        NodeCreator::instance().addItem("ConcatBlock",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("ConcatBlock",QColor(0x954f75));

        NodeCreator::instance().addItem("Connected",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Connected",QColor(0x009394));

        NodeCreator::instance().addItem("Conv",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"filters", "string", AttributeInfo::Type::member},
            {"kernel", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"pad", "string", AttributeInfo::Type::member},
            {"dilate", "string", AttributeInfo::Type::member},
            {"group", "string", AttributeInfo::Type::member},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Conv",QColor(0x49D3D6));

        NodeCreator::instance().addItem("Slice",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"start0", "string", AttributeInfo::Type::member},
            {"step0", "string", AttributeInfo::Type::member},
            {"start1", "string", AttributeInfo::Type::member},
            {"step1", "string", AttributeInfo::Type::member},
            {"start2", "string", AttributeInfo::Type::member},
            {"step2", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Slice",QColor(0xd75093));

        NodeCreator::instance().addItem("DeConv",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"filters", "string", AttributeInfo::Type::member},
            {"kernel", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"pad", "string", AttributeInfo::Type::member},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("DeConv",QColor(0x27F2BD));


        NodeCreator::instance().addItem("ConvBN",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"filters", "string", AttributeInfo::Type::member},
            {"kernel", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"pad", "string", AttributeInfo::Type::member},
            {"dilate", "string", AttributeInfo::Type::member},
            {"group", "string", AttributeInfo::Type::member},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("ConvBN",QColor(0xF14BBB));

        NodeCreator::instance().addItem("ConvDW",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"filters", "string", AttributeInfo::Type::member},
            {"kernel", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"pad", "string", AttributeInfo::Type::member},
            {"dilate", "string", AttributeInfo::Type::member},
            {"group", "string", AttributeInfo::Type::member},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("ConvDW",QColor(0xFF5186));

        NodeCreator::instance().addItem("Crop",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Crop",QColor(0xA83A90));

        NodeCreator::instance().addItem("LocalAvgPool",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"filters", "string", AttributeInfo::Type::member},
            {"kernel", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"pad", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("LocalAvgPool",QColor(0xFFDB97));

        NodeCreator::instance().addItem("GlobalAvgPool",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("GlobalAvgPool",QColor(0xFFDB97));

        NodeCreator::instance().addItem("MaxPool",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"filters", "string", AttributeInfo::Type::member},
            {"kernel", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"pad", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("MaxPool",QColor(0x3353CA));


        NodeCreator::instance().addItem("Padding",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"pad", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Padding",QColor(0xFFAA5A));

        NodeCreator::instance().addItem("Permute",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"dim0", "string", AttributeInfo::Type::member},
            {"dim1", "string", AttributeInfo::Type::member},
            {"dim2", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Permute",QColor(0xaf2178));

        NodeCreator::instance().addItem("View",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("View",QColor(0x1f4aa3));

        NodeCreator::instance().addItem("Reduction",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"type", "string", AttributeInfo::Type::member},
            {"axis", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Reduction",QColor(0xe0710f));

        NodeCreator::instance().addItem("VarOp",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"type", "string", AttributeInfo::Type::member},
            {"layer", "string", AttributeInfo::Type::member},
            {"const", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("VarOp",QColor(0xff4e1e));

        NodeCreator::instance().addItem("Res2Block",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Res2Block",QColor(0x6DACFF));

        NodeCreator::instance().addItem("ResBlock",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("ResBlock",QColor(0xFF5B74));

        NodeCreator::instance().addItem("Route",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"group", "string", AttributeInfo::Type::member},
            {"type", "string", AttributeInfo::Type::member},
            {"act", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Route",QColor(0xBB87FF));

        NodeCreator::instance().addItem("SoftMax",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"groups", "string", AttributeInfo::Type::member},
            {"temperature", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("SoftMax",QColor(0x60FFDF));

        NodeCreator::instance().addItem("UpSample",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"type", "string", AttributeInfo::Type::member},
            {"stride", "string", AttributeInfo::Type::member},
            {"scale", "string", AttributeInfo::Type::member},
            {"algin", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("UpSample",QColor(0x886BFF));

        NodeCreator::instance().addItem("Yolo",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"classes", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"yolo", "string", AttributeInfo::Type::member},
            {"output", "string", AttributeInfo::Type::output},
        });
        ColorTabel::instance().addColor("Yolo",QColor(0xFF5000));

        NodeCreator::instance().addItem("YoloOut",
        {
            {"input", "string", AttributeInfo::Type::input},
            {"conf", "string", AttributeInfo::Type::member},
            {"nms", "string", AttributeInfo::Type::member},
            {"inplace", "string", AttributeInfo::Type::member},
            {"yolo", "string", AttributeInfo::Type::member}

        });
        ColorTabel::instance().addColor("YoloOut",QColor(0x7A85FF));
    }
};



}
#endif // MSNHVIEWERNODECFG_H
