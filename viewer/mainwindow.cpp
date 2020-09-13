#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <MsnhViewerNode.h>
#include <MsnhViewerNodeCreator.h>
#include <MsnhViewerThemeManager.h>
#include <QPrinter>
#include <QPrintDialog>
#include <MsnhViewerNodeCfg.h>


using namespace MsnhViewer;
QPlainTextEdit *MainWindow::logger = nullptr;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    scene = new MsnhViewer::Scene(this);
    subScene = new MsnhViewer::Scene(this);

    builder.setPreviewMode(true);

    logger = ui->logger;

    timer = new QTimer(this);
    connect(timer,&QTimer::timeout,this,&MainWindow::doTimer);
    timer->start(100);

    NodeCfg::configNodes();

    ui->graphicsView->setScene(scene);
    ui->graphicsView2->setScene(subScene);

    progressBar = new QProgressBar(this);
    progressBar->setMinimum(0);
    progressBar->setMaximum(100);
    progressBar->setMaximumWidth(200);
    progressBar->setMinimumWidth(200);
    progressBar->hide();

    ui->statusBar->addWidget(progressBar);
}

MainWindow::~MainWindow()
{
    timer->stop();
    delete ui;
}

void MainWindow::on_actionOpen_triggered()
{
    QString filePath=QFileDialog::getOpenFileName(this,"choose file","","msnhnet(*.msnhnet)");

    if(filePath == "")
    {
        qWarning()<<"File path can not be empty";
        return;
    }

    scene->clear();
    subScene->clear();

    try
    {
        builder.buildNetFromMsnhNet(filePath.toLocal8Bit().toStdString());
        std::cout<<(builder.getLayerDetail());
    }
    catch(Msnhnet::Exception &ex)
    {
        qWarning()<<QString::fromStdString(ex.what())<<QString::fromStdString(ex.getErrFile())
                 <<QString::number(ex.getErrLine());
        return;
    }

    qInfo()<<"Loading.... " + filePath;

    Node* node1 = NodeCreator::instance().createItem("Inputs", "0", {0,0});
    Msnhnet::BaseLayer *layer = builder.getNet()->layers[0];
    node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel()));
    scene->addNode(node1);

    qreal width       =   0;
    qreal height      =   0;
    qreal finalHeight =   -1;

    ui->graphicsView->fitView();

    int layerNum        =   builder.getNet()->layers.size();

    startProgressBar();

    for (int i = 0; i < layerNum; ++i)
    {
        updateProgressBar(static_cast<int>(100.f*i/layerNum));

        width =  (i+1)*256;
        QString layerName = QString::fromStdString(builder.getNet()->layers[i]->getLayerName()).trimmed();

        if(layerName == "Route")
        {
            height = height + 400;
        }

        if(layerName == "VarOp")
        {
            Msnhnet::VariableOpLayer *layer = reinterpret_cast<Msnhnet::VariableOpLayer *>(builder.getNet()->layers[i]);
            if(layer->getInputLayerIndexes().size()==1)
            {
                height = height + 400;
            }

        }

        if(layerName == "Yolo" && finalHeight==-1)
        {
            height = height + 400;
            finalHeight = height;
        }

        if(layerName == "YoloOut")
        {
            height = finalHeight;
        }

        Node* node = NodeCreator::instance().createItem(layerName, QString::number(i+1), {width,height});

        if(layerName == "Conv" || layerName == "ConvBN" || layerName == "ConvDW")
        {
            Msnhnet::ConvolutionalLayer *layer = reinterpret_cast<Msnhnet::ConvolutionalLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString filters     = QString("%1").arg(layer->getOutChannel());
            QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
            QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
            QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
            QString dilation    = QString("%1*%2").arg(layer->getDilationX()).arg(layer->getDilationY());
            QString group       = QString("%1/%2").arg(layer->getGroupIndex()).arg(layer->getGroups());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(filters   );
            node->attributes()[2]->setData(kernel    );
            node->attributes()[3]->setData(stride    );
            node->attributes()[4]->setData(padding   );
            node->attributes()[5]->setData(dilation  );
            node->attributes()[6]->setData(group     );
            node->attributes()[7]->setData(activation);
            node->attributes()[8]->setData(inplace   );
            node->attributes()[9]->setData(output    );
        }

        if(layerName == "DeConv")
        {
            Msnhnet::DeConvolutionalLayer *layer = reinterpret_cast<Msnhnet::DeConvolutionalLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString filters     = QString("%1").arg(layer->getOutChannel());
            QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
            QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
            QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(filters   );
            node->attributes()[2]->setData(kernel    );
            node->attributes()[3]->setData(stride    );
            node->attributes()[4]->setData(padding   );
            node->attributes()[5]->setData(activation);
            node->attributes()[6]->setData(inplace   );
            node->attributes()[7]->setData(output    );
        }

        if(layerName == "Connected")
        {
            Msnhnet::ConnectedLayer *layer = reinterpret_cast<Msnhnet::ConnectedLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(inplace   );
            node->attributes()[3]->setData(output    );
        }

        if(layerName == "MaxPool")
        {
            Msnhnet::MaxPoolLayer *layer = reinterpret_cast<Msnhnet::MaxPoolLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString filters     = QString("%1").arg(layer->getOutChannel());
            QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
            QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
            QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(filters);
            node->attributes()[2]->setData(kernel );
            node->attributes()[3]->setData(stride );
            node->attributes()[4]->setData(padding);
            node->attributes()[5]->setData(inplace);
            node->attributes()[6]->setData(output );
        }

        if(layerName == "GlobalAvgPool")
        {
            Msnhnet::GlobalAvgPoolLayer *layer = reinterpret_cast<Msnhnet::GlobalAvgPoolLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(inplace);
            node->attributes()[2]->setData(output );
        }

        if(layerName == "LocalAvgPool")
        {
            Msnhnet::LocalAvgPoolLayer *layer = reinterpret_cast<Msnhnet::LocalAvgPoolLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString filters     = QString("%1").arg(layer->getOutChannel());
            QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
            QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
            QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(filters);
            node->attributes()[2]->setData(kernel );
            node->attributes()[3]->setData(stride );
            node->attributes()[4]->setData(padding);
            node->attributes()[5]->setData(inplace);
            node->attributes()[6]->setData(output );
        }

        if(layerName == "Empty")
        {
            Msnhnet::EmptyLayer *layer = reinterpret_cast<Msnhnet::EmptyLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(inplace);
            node->attributes()[2]->setData(output );
        }

        if(layerName == "Activate")
        {
            Msnhnet::ActivationLayer *layer = reinterpret_cast<Msnhnet::ActivationLayer *>(builder.getNet()->layers[i]);
            QString input           = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString act             = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace         = QString("%1").arg(layer->getMemReUse());
            QString output          = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(act    );
            node->attributes()[2]->setData(inplace);
            node->attributes()[3]->setData(output );
        }

        if(layerName == "BatchNorm")
        {
            Msnhnet::BatchNormLayer *layer = reinterpret_cast<Msnhnet::BatchNormLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(inplace   );
            node->attributes()[3]->setData(output    );
        }


        if(layerName == "AddBlock")
        {
            Msnhnet::AddBlockLayer *layer = reinterpret_cast<Msnhnet::AddBlockLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(inplace   );
            node->attributes()[3]->setData(output    );
        }

        if(layerName == "ConcatBlock")
        {
            Msnhnet::ConcatBlockLayer *layer = reinterpret_cast<Msnhnet::ConcatBlockLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(inplace   );
            node->attributes()[3]->setData(output    );
        }

        if(layerName == "Res2Block")
        {
            Msnhnet::Res2BlockLayer *layer = reinterpret_cast<Msnhnet::Res2BlockLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(inplace   );
            node->attributes()[3]->setData(output    );
        }

        if(layerName == "ResBlock")
        {
            Msnhnet::ResBlockLayer *layer = reinterpret_cast<Msnhnet::ResBlockLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(inplace   );
            node->attributes()[3]->setData(output    );
        }

        if(layerName == "Crop")
        {
            Msnhnet::CropLayer *layer = reinterpret_cast<Msnhnet::CropLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(inplace);
            node->attributes()[2]->setData(output );
        }

        if(layerName == "Padding")
        {
            Msnhnet::PaddingLayer *layer = reinterpret_cast<Msnhnet::PaddingLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString padding     = QString("%1/%2/%3/%4").arg(layer->getTop()).arg(layer->getDown()).arg(layer->getLeft()).arg(layer->getRight());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(padding);
            node->attributes()[2]->setData(inplace);
            node->attributes()[3]->setData(output );
        }

        if(layerName == "Reduction")
        {
            Msnhnet::ReductionLayer *layer = reinterpret_cast<Msnhnet::ReductionLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString type        = (layer->getReductionType()==ReductionType::REDUCTION_SUM)?"sum":"mean";
            QString axis        = QString::number(layer->getAxis());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(type   );
            node->attributes()[2]->setData(axis   );
            node->attributes()[3]->setData(inplace);
            node->attributes()[4]->setData(output );
        }

        if(layerName == "Permute")
        {
            Msnhnet::PermuteLayer *layer = reinterpret_cast<Msnhnet::PermuteLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString dim0        = QString::number(layer->getDim0());
            QString dim1        = QString::number(layer->getDim1());
            QString dim2        = QString::number(layer->getDim2());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(dim0   );
            node->attributes()[2]->setData(dim1   );
            node->attributes()[3]->setData(dim2   );
            node->attributes()[4]->setData(inplace);
            node->attributes()[5]->setData(output );
        }

        if(layerName == "View")
        {
            Msnhnet::ViewLayer *layer = reinterpret_cast<Msnhnet::ViewLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(inplace);
            node->attributes()[2]->setData(output );
        }


        if(layerName == "VarOp")
        {
            Msnhnet::VariableOpLayer *layer = reinterpret_cast<Msnhnet::VariableOpLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString index       = (layer->getInputLayerIndexes().size()==1)?QString::number(layer->getInputLayerIndexes()[0]+1):"-----";
            QString layerType   = QString::fromStdString(Msnhnet::VariableOpParams::getStrFromVarOpType(layer->getVarOpType()));
            QString constVal    = QString::number(layer->getConstVal());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input    );
            node->attributes()[1]->setData(layerType);
            node->attributes()[2]->setData(index    );
            node->attributes()[3]->setData(constVal );
            node->attributes()[4]->setData(inplace  );
            node->attributes()[5]->setData(output   );
        }

        if(layerName == "Slice")
        {
            Msnhnet::SliceLayer *layer = reinterpret_cast<Msnhnet::SliceLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString start0      = QString("%1").arg(layer->getStart0());
            QString step0       = QString("%1").arg(layer->getStep0());
            QString start1      = QString("%1").arg(layer->getStart1());
            QString step1       = QString("%1").arg(layer->getStep1());
            QString start2      = QString("%1").arg(layer->getStart2());
            QString step2       = QString("%1").arg(layer->getStep2());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input    );
            node->attributes()[1]->setData(start0   );
            node->attributes()[2]->setData(step0    );
            node->attributes()[3]->setData(start1   );
            node->attributes()[4]->setData(step1    );
            node->attributes()[5]->setData(start2   );
            node->attributes()[6]->setData(step2    );
            node->attributes()[7]->setData(inplace  );
            node->attributes()[8]->setData(output   );
        }

        if(layerName == "Route")
        {
            Msnhnet::RouteLayer *layer = reinterpret_cast<Msnhnet::RouteLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("---------");
            QString group       = QString("%1/%2").arg(layer->getGroupIndex()).arg(layer->getGroups());
            QString type        = (layer->getAddModel() == 1)?"ADD":"Connect";
            QString act         = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(group  );
            node->attributes()[2]->setData(type   );
            node->attributes()[3]->setData(act    );
            node->attributes()[4]->setData(inplace);
            node->attributes()[5]->setData(output );
        }

        if(layerName == "SoftMax")
        {
            Msnhnet::SoftMaxLayer *layer = reinterpret_cast<Msnhnet::SoftMaxLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1").arg(layer->getInputNum());
            QString groups      = QString("%1").arg(layer->getGroups());
            QString temperature = QString("%1").arg(layer->getTemperature());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1").arg(layer->getOutputNum());

            node->attributes()[0]->setData(input      );
            node->attributes()[1]->setData(groups     );
            node->attributes()[2]->setData(temperature);
            node->attributes()[3]->setData(inplace    );
            node->attributes()[4]->setData(output     );
        }

        if(layerName == "UpSample")
        {
            Msnhnet::UpSampleLayer *layer = reinterpret_cast<Msnhnet::UpSampleLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString type        = QString::fromStdString(Msnhnet::UpSampleParams::getStrFromUnsampleType(layer->getUpsampleType()));
            QString scale       = QString("%1*%2").arg(int(layer->getScaleX()*1000)/1000.0f).arg(int(layer->getScaleY()*1000)/1000.0f);
            QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
            QString alignCorner = QString("%1").arg(layer->getAlignCorners());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input      );
            node->attributes()[1]->setData(type       );
            node->attributes()[2]->setData(stride     );
            node->attributes()[3]->setData(scale      );
            node->attributes()[4]->setData(alignCorner);
            node->attributes()[5]->setData(inplace    );
            node->attributes()[6]->setData(output     );
        }

        if(layerName == "Yolo")
        {
            Msnhnet::YoloLayer *layer = reinterpret_cast<Msnhnet::YoloLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString classes     = QString("%1").arg(layer->getClassNum());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString yoloType    = QString::fromStdString(Msnhnet::getStrFromYoloType(layer->getYoloType()));
            QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

            node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(classes);
            node->attributes()[2]->setData(inplace);
            node->attributes()[3]->setData(yoloType);
            node->attributes()[4]->setData(output);
        }

        if(layerName == "YoloOut")
        {
            Msnhnet::YoloOutLayer *layer = reinterpret_cast<Msnhnet::YoloOutLayer *>(builder.getNet()->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
            QString conf        = QString("%1").arg(layer->getConfThresh());
            QString nms         = QString("%1").arg(layer->getNmsThresh());
            QString inplace     = QString("%1").arg(layer->getMemReUse());
            QString yoloType    = QString::fromStdString(Msnhnet::getStrFromYoloType(layer->getYoloType()));

            node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(conf);
            node->attributes()[2]->setData(nms);
            node->attributes()[3]->setData(inplace);
            node->attributes()[4]->setData(yoloType);
        }

        if(node == nullptr)
        {
            scene->clear();
            return;
        }

        scene->addNode(node);

    }

    scene->setSceneH(height);
    scene->setSceneW(width);

    for (int i = 0; i < layerNum; ++i)
    {
        updateProgressBar(static_cast<int>(100.f*i/layerNum));
        if(i>0)
        {
            //      => name i+1
            QString layerName = QString::fromStdString(builder.getNet()->layers[i]->getLayerName()).trimmed();

            if(layerName == "Route")
            {
                Msnhnet::RouteLayer *layer  = reinterpret_cast<Msnhnet::RouteLayer *>(builder.getNet()->layers[i]);
                std::vector<int>    indexes = layer->getInputLayerIndexes();

                for (int j = 0; j < indexes.size(); ++j)
                {
                    scene->connectNode(QString::number(indexes[j]+1),"output",QString::number(i+1),"input");
                }
            }
            else if(layerName == "YoloOut")
            {
                Msnhnet::YoloOutLayer *layer  = reinterpret_cast<Msnhnet::YoloOutLayer *>(builder.getNet()->layers[i]);
                std::vector<int>    indexes = layer->getYoloIndexes();

                for (int j = 0; j < indexes.size(); ++j)
                {
                    scene->connectNode(QString::number(indexes[j]+1),"output",QString::number(i+1),"input");
                }
            }
            else if(layerName == "VarOp")
            {
                scene->connectNode(QString::number(i),"output",QString::number(i+1),"input");
                Msnhnet::VariableOpLayer *layer  = reinterpret_cast<Msnhnet::VariableOpLayer *>(builder.getNet()->layers[i]);
                std::vector<int>    indexes = layer->getInputLayerIndexes();

                for (int j = 0; j < indexes.size(); ++j)
                {
                    scene->connectNode(QString::number(indexes[j]+1),"output",QString::number(i+1),"input");
                }
            }
            else
            {
                scene->connectNode(QString::number(i),"output",QString::number(i+1),"input");
            }
        }
        else
        {
            scene->connectNode(QString::number(i),"output",QString::number(i+1),"input");
        }

    }

    stopProgressBar();

    scene->update();

    qInfo()<<"Load Done. " + filePath + "";

}

void MainWindow::on_actionPrint_triggered()
{


    QString filePath=QFileDialog::getSaveFileName(this,"save file","","svg(*.svg)");

    if(filePath == "")
    {
        qWarning()<<"File path can not be empty";
        return;
    }

    QSvgGenerator svgGen;
    svgGen.setFileName(filePath);
    svgGen.setSize({(int)scene->getSceneW(),(int)scene->getSceneH()});
    svgGen.setViewBox(QRect(0,0,(int)scene->getSceneW(),(int)scene->getSceneH()));
    svgGen.setFileName(filePath);
    QPainter painter(&svgGen);
    painter.setRenderHint(QPainter::Antialiasing);
    ui->graphicsView->scene()->render(&painter);

    qInfo()<<"Export svg Done.  " + filePath;

}

void MainWindow::on_actionHand_triggered()
{
    if(ui->actionHand->isChecked())
    {
        ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
        ui->graphicsView2->setDragMode(QGraphicsView::ScrollHandDrag);
    }
    else
    {
        ui->graphicsView->setDragMode(QGraphicsView::RubberBandDrag);
        ui->graphicsView2->setDragMode(QGraphicsView::RubberBandDrag);
    }
}

void MainWindow::doTimer()
{
    if(NodeSelect::selectNode != "null")
    {
        QString clickedNode     = NodeSelect::selectNode;

        int clickedIndex        = clickedNode.toInt();

        NodeSelect::selectNode  = "null";

        QString nodeType        = scene->nodes()[clickedIndex]->nodeType();

        if(nodeType == "ConcatBlock")
        {
            subScene->clear();

            QString inNode  = clickedNode +"_I";
            QString outNode = clickedNode +"_O";

            // inputs
            Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

            Msnhnet::ConcatBlockLayer *layer = reinterpret_cast<Msnhnet::ConcatBlockLayer*>(builder.getNet()->layers[clickedIndex-1]);

            node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel()));
            subScene->addNode(node1);

            std::vector<std::vector<Msnhnet::BaseLayer *>> branchLayers = layer->branchLayers;

            qreal widthMax    =   0;

            for (int i = 0; i <branchLayers.size(); ++i)
            {
                qreal tmp     =   (branchLayers[i].size()+1)*256  ;
                if(widthMax < tmp)
                {
                    widthMax = tmp;
                }
            }

            //outputs
            Node* node2 = NodeCreator::instance().createItem("ConcatOutputs", outNode, {widthMax,0});

            node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel()));
            subScene->addNode(node2);

            qreal width       =   256;
            qreal height      =   0;


            // concatblock per branch
            for (int i = 0; i <branchLayers.size(); ++i)
            {
                std::vector<Msnhnet::BaseLayer *> branch = branchLayers[i];

                for (int j = 0; j < branch.size(); ++j)
                {
                    QString layerName = QString::fromStdString( branch[j]->getLayerName()).trimmed();

                    QString nodeName  = clickedNode + "_" + QString::number(i) + "_" + QString::number(j);

                    Node* node = NodeCreator::instance().createItem(layerName, nodeName , {width,height});

                    createNode(node, layerName, branch[j]);

                    if(node == nullptr)
                    {
                        subScene->clear();
                        return;
                    }

                    width += 256;

                    subScene->addNode(node);

                    QString lastNodeName  = clickedNode + "_" + QString::number(i) + "_" + QString::number(j-1);
                    if(j==0)
                    {
                        subScene->connectNode(inNode,"output",nodeName,"input");
                    }
                    else
                    {
                        subScene->connectNode(lastNodeName,"output",nodeName,"input");
                    }

                    if(j == branch.size()-1)
                    {
                        subScene->connectNode(nodeName,"output",outNode,"input");
                    }
                }

                width = 256;
                height += 400;
            }
        }
        else if(nodeType == "AddBlock")
        {
            subScene->clear();

            QString inNode  = clickedNode +"_I";
            QString outNode = clickedNode +"_O";

            // inputs
            Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

            Msnhnet::AddBlockLayer *layer = reinterpret_cast<Msnhnet::AddBlockLayer*>(builder.getNet()->layers[clickedIndex-1]);

            node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel()));
            subScene->addNode(node1);

            std::vector<std::vector<Msnhnet::BaseLayer *>> branchLayers = layer->branchLayers;

            qreal widthMax    =   0;

            for (int i = 0; i <branchLayers.size(); ++i)
            {
                qreal tmp     =   (branchLayers[i].size()+1)*256  ;
                if(widthMax < tmp)
                {
                    widthMax = tmp;
                }
            }

            //outputs
            Node* node2 = NodeCreator::instance().createItem("AddOutputs", outNode, {widthMax,0});

            node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel()));
            subScene->addNode(node2);

            qreal width       =   256;
            qreal height      =   0;

            // addblock per branch
            for (int i = 0; i <branchLayers.size(); ++i)
            {
                std::vector<Msnhnet::BaseLayer *> branch = branchLayers[i];

                for (int j = 0; j < branch.size(); ++j)
                {
                    QString layerName = QString::fromStdString( branch[j]->getLayerName()).trimmed();

                    QString nodeName  = clickedNode + "_" + QString::number(i) + "_" + QString::number(j);

                    Node* node = NodeCreator::instance().createItem(layerName, nodeName , {width,height});

                    if(node == nullptr)
                    {
                        subScene->clear();
                        return;
                    }

                    createNode(node, layerName, branch[j]);

                    width += 256;

                    subScene->addNode(node);

                    QString lastNodeName  = clickedNode + "_" + QString::number(i) + "_" + QString::number(j-1);
                    if(j==0)
                    {
                        subScene->connectNode(inNode,"output",nodeName,"input");
                    }
                    else
                    {
                        subScene->connectNode(lastNodeName,"output",nodeName,"input");
                    }

                    if(j == branch.size()-1)
                    {
                        subScene->connectNode(nodeName,"output",outNode,"input");
                    }
                }

                width = 256;
                height += 400;
            }
        }
        else if(nodeType == "Res2Block")
        {
            subScene->clear();

            QString inNode  = clickedNode +"_I";
            QString outNode = clickedNode +"_O";

            // inputs
            Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

            Msnhnet::Res2BlockLayer *layer = reinterpret_cast<Msnhnet::Res2BlockLayer*>(builder.getNet()->layers[clickedIndex-1]);

            node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel()));
            subScene->addNode(node1);

            std::vector<Msnhnet::BaseLayer *>  branchLayers = layer->branchLayers;
            std::vector<Msnhnet::BaseLayer *>  baseLayers   = layer->baseLayers;

            qreal widthMax    =   0;

            qreal tmp     =   (branchLayers.size()+1)*256  ;
            if(widthMax < tmp)
            {
                widthMax = tmp;
            }

            tmp     =   (baseLayers.size()+1)*256 ;
            if(widthMax < tmp)
            {
                widthMax = tmp;
            }


            //outputs
            Node* node2 = NodeCreator::instance().createItem("AddOutputs", outNode, {widthMax,0});

            node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel()));
            subScene->addNode(node2);

            qreal width       =   256;
            qreal height      =   0;

            // base of res2block
            for (int j = 0; j < baseLayers.size(); ++j)
            {
                QString layerName = QString::fromStdString( baseLayers[j]->getLayerName()).trimmed();

                QString nodeName  = clickedNode + "_0_" + QString::number(j);

                Node* node = NodeCreator::instance().createItem(layerName, nodeName , {width,height});

                if(node == nullptr)
                {
                    subScene->clear();
                    return;
                }

                createNode(node, layerName, baseLayers[j]);

                width += 256;

                subScene->addNode(node);

                QString lastNodeName  = clickedNode + "_0_" + QString::number(j-1);
                if(j==0)
                {
                    subScene->connectNode(inNode,"output",nodeName,"input");
                }
                else
                {
                    subScene->connectNode(lastNodeName,"output",nodeName,"input");
                }

                if(j == baseLayers.size()-1)
                {
                    subScene->connectNode(nodeName,"output",outNode,"input");
                }

            }

            width = 256;
            height += 400;

            // branch of res2block
            for (int j = 0; j < branchLayers.size(); ++j)
            {
                QString layerName = QString::fromStdString( branchLayers[j]->getLayerName()).trimmed();

                QString nodeName  = clickedNode + "_1_" + QString::number(j);

                Node* node = NodeCreator::instance().createItem(layerName, nodeName , {width,height});

                if(node == nullptr)
                {
                    subScene->clear();
                    return;
                }

                createNode(node, layerName, branchLayers[j]);


                width += 256;

                subScene->addNode(node);

                QString lastNodeName  = clickedNode + "_1_" + QString::number(j-1);
                if(j==0)
                {
                    subScene->connectNode(inNode,"output",nodeName,"input");
                }
                else
                {
                    subScene->connectNode(lastNodeName,"output",nodeName,"input");
                }

                if(j == branchLayers.size()-1)
                {
                    subScene->connectNode(nodeName,"output",outNode,"input");
                }
            }

        }
        else if(nodeType == "ResBlock")
        {
            subScene->clear();

            QString inNode  = clickedNode +"_I";
            QString outNode = clickedNode +"_O";

            // inputs
            Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

            Msnhnet::ResBlockLayer *layer = reinterpret_cast<Msnhnet::ResBlockLayer*>(builder.getNet()->layers[clickedIndex-1]);

            node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel()));

            subScene->addNode(node1);

            std::vector<Msnhnet::BaseLayer *>  baseLayers   = layer->baseLayers;

            qreal widthMax    =   0;

            qreal tmp     =   (baseLayers.size()+1)*256  ;
            if(widthMax < tmp)
            {
                widthMax = tmp;
            }


            //outputs
            Node* node2 = NodeCreator::instance().createItem("AddOutputs", outNode, {widthMax,0});

            node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel()));
            subScene->addNode(node2);

            subScene->connectNode(inNode,"output",outNode,"input");

            qreal width       =   256;
            qreal height      =   400;

            // base of resblock
            for (int j = 0; j < baseLayers.size(); ++j)
            {
                QString layerName = QString::fromStdString( baseLayers[j]->getLayerName()).trimmed();

                QString nodeName  = clickedNode + "_0_" + QString::number(j);

                Node* node = NodeCreator::instance().createItem(layerName, nodeName , {width,height});

                if(node == nullptr)
                {
                    subScene->clear();
                    return;
                }

                createNode(node, layerName, baseLayers[j]);

                width += 256;

                subScene->addNode(node);

                QString lastNodeName  = clickedNode + "_0_" + QString::number(j-1);
                if(j==0)
                {
                    subScene->connectNode(inNode,"output",nodeName,"input");
                }
                else
                {
                    subScene->connectNode(lastNodeName,"output",nodeName,"input");
                }

                if(j == baseLayers.size()-1)
                {
                    subScene->connectNode(nodeName,"output",outNode,"input");
                }

            }

        }
        else
        {
            subScene->clear();
        }

        qDebug()<<"Expand "<<clickedNode;
        ui->graphicsView2->fitView();
    }
}

void MainWindow::createNode(Node *node, const QString &layerName, Msnhnet::BaseLayer *inLayer)
{
    if(layerName == "Conv" || layerName == "ConvBN" || layerName == "ConvDW")
    {
        Msnhnet::ConvolutionalLayer *layer = reinterpret_cast<Msnhnet::ConvolutionalLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString filters     = QString("%1").arg(layer->getOutChannel());
        QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
        QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
        QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
        QString dilation    = QString("%1*%2").arg(layer->getDilationX()).arg(layer->getDilationY());
        QString group       = QString("%1/%2").arg(layer->getGroupIndex()).arg(layer->getGroups());
        QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input     );
        node->attributes()[1]->setData(filters   );
        node->attributes()[2]->setData(kernel    );
        node->attributes()[3]->setData(stride    );
        node->attributes()[4]->setData(padding   );
        node->attributes()[5]->setData(dilation  );
        node->attributes()[6]->setData(group     );
        node->attributes()[7]->setData(activation);
        node->attributes()[8]->setData(inplace   );
        node->attributes()[9]->setData(output    );
    }

    if(layerName == "Connected")
    {
        Msnhnet::ConnectedLayer *layer = reinterpret_cast<Msnhnet::ConnectedLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input     );
        node->attributes()[1]->setData(activation);
        node->attributes()[2]->setData(inplace   );
        node->attributes()[3]->setData(output    );
    }

    if(layerName == "MaxPool")
    {
        Msnhnet::MaxPoolLayer *layer = reinterpret_cast<Msnhnet::MaxPoolLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString filters     = QString("%1").arg(layer->getOutChannel());
        QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
        QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
        QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(filters);
        node->attributes()[2]->setData(kernel );
        node->attributes()[3]->setData(stride );
        node->attributes()[4]->setData(padding);
        node->attributes()[5]->setData(inplace);
        node->attributes()[6]->setData(output );
    }

    if(layerName == "LocalAvgPool")
    {
        Msnhnet::LocalAvgPoolLayer *layer = reinterpret_cast<Msnhnet::LocalAvgPoolLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString filters     = QString("%1").arg(layer->getOutChannel());
        QString kernel      = QString("%1*%2").arg(layer->getKSizeX()).arg(layer->getKSizeY());
        QString stride      = QString("%1*%2").arg(layer->getStrideX()).arg(layer->getStrideY());
        QString padding     = QString("%1*%2").arg(layer->getPaddingX()).arg(layer->getPaddingY());
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(filters);
        node->attributes()[2]->setData(kernel );
        node->attributes()[3]->setData(stride );
        node->attributes()[4]->setData(padding);
        node->attributes()[5]->setData(inplace);
        node->attributes()[6]->setData(output );
    }

    if(layerName == "GlobalAvgPool")
    {
        Msnhnet::GlobalAvgPoolLayer *layer = reinterpret_cast<Msnhnet::GlobalAvgPoolLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(inplace);
        node->attributes()[2]->setData(output );
    }

    if(layerName == "Empty")
    {
        Msnhnet::EmptyLayer *layer = reinterpret_cast<Msnhnet::EmptyLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(inplace);
        node->attributes()[2]->setData(output );
    }

    if(layerName == "BatchNorm")
    {
        Msnhnet::BatchNormLayer *layer = reinterpret_cast<Msnhnet::BatchNormLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->getActivation()));
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input     );
        node->attributes()[1]->setData(activation);
        node->attributes()[2]->setData(inplace   );
        node->attributes()[3]->setData(output    );
    }

    if(layerName == "Padding")
    {
        Msnhnet::PaddingLayer *layer = reinterpret_cast<Msnhnet::PaddingLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->getWidth()).arg(layer->getHeight()).arg(layer->getChannel());
        QString padding     = QString("%1/%2/%3/%4").arg(layer->getTop()).arg(layer->getDown()).arg(layer->getLeft()).arg(layer->getRight());
        QString inplace     = QString("%1").arg(layer->getMemReUse());
        QString output      = QString("%1*%2*%3").arg(layer->getOutWidth()).arg(layer->getOutHeight()).arg(layer->getOutChannel());

        node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(padding);
        node->attributes()[2]->setData(inplace);
        node->attributes()[3]->setData(output );
    }
}

void MainWindow::on_actionPowerOff_triggered()
{
    qApp->quit();
}

void MainWindow::startProgressBar()
{
    progressBar->show();
    progressBar->setValue(0);
}

void MainWindow::stopProgressBar()
{
    progressBar->hide();
    progressBar->setValue(0);
}

void MainWindow::updateProgressBar(const int &val)
{
    progressBar->setValue(val);
    QApplication::processEvents();
}

