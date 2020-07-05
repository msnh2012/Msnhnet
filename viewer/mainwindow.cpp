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
    Msnhnet::BaseLayer *layer = builder.net->layers[0];
    node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel));
    scene->addNode(node1);

   qreal width       =   0;
    qreal height      =   0;
    qreal finalHeight =   -1;

   for (int i = 0; i < builder.net->layers.size(); ++i)
    {
        width =  (i+1)*256;
        QString layerName = QString::fromStdString(builder.net->layers[i]->layerName).trimmed();

       if(layerName == "Route")
        {
            height = height + 400;
        }

       if(layerName == "Yolov3" && finalHeight==-1)
        {
            height = height + 400;
            finalHeight = height;
        }

       if(layerName == "Yolov3Out")
        {
            height = finalHeight;
        }

       Node* node = NodeCreator::instance().createItem(layerName, QString::number(i+1), {width,height});

       if(layerName == "Conv" || layerName == "ConvBN" || layerName == "ConvDW")
        {
            Msnhnet::ConvolutionalLayer *layer = reinterpret_cast<Msnhnet::ConvolutionalLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString filters     = QString("%1").arg(layer->outChannel);
            QString kernel      = QString("%1*%2").arg(layer->kSizeX).arg(layer->kSizeY);
            QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
            QString padding     = QString("%1*%2").arg(layer->paddingX).arg(layer->paddingY);
            QString dilation    = QString("%1*%2").arg(layer->dilationX).arg(layer->dilationY);
            QString group       = QString("%1/%2").arg(layer->groupIndex).arg(layer->groups);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(filters   );
            node->attributes()[2]->setData(kernel    );
            node->attributes()[3]->setData(stride    );
            node->attributes()[4]->setData(padding   );
            node->attributes()[5]->setData(dilation  );
            node->attributes()[6]->setData(group     );
            node->attributes()[7]->setData(activation);
            node->attributes()[8]->setData(output    );
        }

       if(layerName == "Connected")
        {
            Msnhnet::ConnectedLayer *layer = reinterpret_cast<Msnhnet::ConnectedLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(output    );
        }

       if(layerName == "MaxPool")
        {
            Msnhnet::MaxPoolLayer *layer = reinterpret_cast<Msnhnet::MaxPoolLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString filters     = QString("%1").arg(layer->outChannel);
            QString kernel      = QString("%1*%2").arg(layer->kSizeX).arg(layer->kSizeY);
            QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
            QString padding     = QString("%1*%2").arg(layer->paddingX).arg(layer->paddingY);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(filters);
            node->attributes()[2]->setData(kernel );
            node->attributes()[3]->setData(stride );
            node->attributes()[4]->setData(padding);
            node->attributes()[5]->setData(output );
        }

       if(layerName == "LocalAvgPool")
        {
            Msnhnet::LocalAvgPoolLayer *layer = reinterpret_cast<Msnhnet::LocalAvgPoolLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString filters     = QString("%1").arg(layer->outChannel);
            QString kernel      = QString("%1*%2").arg(layer->kSizeX).arg(layer->kSizeY);
            QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
            QString padding     = QString("%1*%2").arg(layer->paddingX).arg(layer->paddingY);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(filters);
            node->attributes()[2]->setData(kernel );
            node->attributes()[3]->setData(stride );
            node->attributes()[4]->setData(padding);
            node->attributes()[5]->setData(output );
        }

       if(layerName == "Empty")
        {
            Msnhnet::EmptyLayer *layer = reinterpret_cast<Msnhnet::EmptyLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input );
            node->attributes()[1]->setData(output);
        }

       if(layerName == "BatchNorm")
        {
            Msnhnet::BatchNormLayer *layer = reinterpret_cast<Msnhnet::BatchNormLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(output    );
        }

       if(layerName == "AddBlock")
        {
            Msnhnet::AddBlockLayer *layer = reinterpret_cast<Msnhnet::AddBlockLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(output    );
        }

       if(layerName == "ConcatBlock")
        {
            Msnhnet::ConcatBlockLayer *layer = reinterpret_cast<Msnhnet::ConcatBlockLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(output    );
        }

       if(layerName == "Res2Block")
        {
            Msnhnet::Res2BlockLayer *layer = reinterpret_cast<Msnhnet::Res2BlockLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(output    );
        }

       if(layerName == "ResBlock")
        {
            Msnhnet::ResBlockLayer *layer = reinterpret_cast<Msnhnet::ResBlockLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input     );
            node->attributes()[1]->setData(activation);
            node->attributes()[2]->setData(output    );
        }

       if(layerName == "Crop")
        {
            Msnhnet::CropLayer *layer = reinterpret_cast<Msnhnet::CropLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input );
            node->attributes()[1]->setData(output);
        }

       if(layerName == "DeConv")
        {

           Msnhnet::DeConvolutionalLayer *layer = reinterpret_cast<Msnhnet::DeConvolutionalLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString filters     = QString("%1").arg(layer->outChannel);
            QString kernel      = QString("%1*%2").arg(layer->kSize).arg(layer->kSize);
            QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input  );
            node->attributes()[1]->setData(filters);
            node->attributes()[2]->setData(kernel );
            node->attributes()[3]->setData(stride );
            node->attributes()[4]->setData(output );
        }

       if(layerName == "Padding")
        {
            Msnhnet::PaddingLayer *layer = reinterpret_cast<Msnhnet::PaddingLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString padding     = QString("%1/%2/%3/%4").arg(layer->top).arg(layer->down).arg(layer->left).arg(layer->right);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(padding);
            node->attributes()[2]->setData(output);
        }

       if(layerName == "Route")
        {
            Msnhnet::RouteLayer *layer = reinterpret_cast<Msnhnet::RouteLayer *>(builder.net->layers[i]);
            QString input       = QString("---------");
            QString group       = QString("%1/%2").arg(layer->groupIndex).arg(layer->groups);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(group);
            node->attributes()[2]->setData(output);
        }

       if(layerName == "SoftMax")
        {
            Msnhnet::SoftMaxLayer *layer = reinterpret_cast<Msnhnet::SoftMaxLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString group       = QString("%1").arg(layer->groups);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(group);
            node->attributes()[2]->setData(output);
        }

       if(layerName == "UpSample")
        {
            Msnhnet::UpSampleLayer *layer = reinterpret_cast<Msnhnet::UpSampleLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString scale       = QString("%1").arg(layer->scale);
            QString stride      = QString("%1*%2").arg(layer->stride).arg(layer->stride);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(scale);
            node->attributes()[2]->setData(stride);
            node->attributes()[3]->setData(output);
        }

       if(layerName == "Yolov3")
        {
            Msnhnet::Yolov3Layer *layer = reinterpret_cast<Msnhnet::Yolov3Layer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString classes     = QString("%1").arg(layer->classNum);
            QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

           node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(classes);
            node->attributes()[2]->setData(output);
        }

       if(layerName == "Yolov3Out")
        {
            Msnhnet::Yolov3OutLayer *layer = reinterpret_cast<Msnhnet::Yolov3OutLayer *>(builder.net->layers[i]);
            QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
            QString conf        = QString("%1").arg(layer->confThresh);
            QString nms         = QString("%1").arg(layer->nmsThresh);

           node->attributes()[0]->setData(input);
            node->attributes()[1]->setData(conf);
            node->attributes()[2]->setData(nms);
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

   for (int i = 0; i < builder.net->layers.size(); ++i)
    {
        if(i>0)
        {

           QString layerName = QString::fromStdString(builder.net->layers[i]->layerName).trimmed();

           if(layerName == "Route")
            {
                Msnhnet::RouteLayer *layer  = reinterpret_cast<Msnhnet::RouteLayer *>(builder.net->layers[i]);
                std::vector<int>    indexes = layer->inputLayerIndexes;

               for (int j = 0; j < indexes.size(); ++j)
                {
                    scene->connectNode(QString::number(indexes[j]+1),"output",QString::number(i+1),"input");
                }
            }
            else if(layerName == "Yolov3Out")
            {
                Msnhnet::Yolov3OutLayer *layer  = reinterpret_cast<Msnhnet::Yolov3OutLayer *>(builder.net->layers[i]);
                std::vector<int>    indexes = layer->yolov3Indexes;

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

   ui->graphicsView->fitView();

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

           Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

           Msnhnet::ConcatBlockLayer *layer = reinterpret_cast<Msnhnet::ConcatBlockLayer*>(builder.net->layers[clickedIndex-1]);

           node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel));
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

           Node* node2 = NodeCreator::instance().createItem("ConcatOutputs", outNode, {widthMax,0});

           node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel));
            subScene->addNode(node2);

           qreal width       =   256;
            qreal height      =   0;

           for (int i = 0; i <branchLayers.size(); ++i)
            {
                std::vector<Msnhnet::BaseLayer *> branch = branchLayers[i];

               for (int j = 0; j < branch.size(); ++j)
                {
                    QString layerName = QString::fromStdString( branch[j]->layerName).trimmed();

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

           Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

           Msnhnet::AddBlockLayer *layer = reinterpret_cast<Msnhnet::AddBlockLayer*>(builder.net->layers[clickedIndex-1]);

           node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel));
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

           Node* node2 = NodeCreator::instance().createItem("AddOutputs", outNode, {widthMax,0});

           node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel));
            subScene->addNode(node2);

           qreal width       =   256;
            qreal height      =   0;

           for (int i = 0; i <branchLayers.size(); ++i)
            {
                std::vector<Msnhnet::BaseLayer *> branch = branchLayers[i];

               for (int j = 0; j < branch.size(); ++j)
                {
                    QString layerName = QString::fromStdString( branch[j]->layerName).trimmed();

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

           Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

           Msnhnet::Res2BlockLayer *layer = reinterpret_cast<Msnhnet::Res2BlockLayer*>(builder.net->layers[clickedIndex-1]);

           node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel));
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

           Node* node2 = NodeCreator::instance().createItem("AddOutputs", outNode, {widthMax,0});

           node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel));
            subScene->addNode(node2);

           qreal width       =   256;
            qreal height      =   0;

           for (int j = 0; j < baseLayers.size(); ++j)
            {
                QString layerName = QString::fromStdString( baseLayers[j]->layerName).trimmed();

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

           for (int j = 0; j < branchLayers.size(); ++j)
            {
                QString layerName = QString::fromStdString( branchLayers[j]->layerName).trimmed();

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

           Node* node1 = NodeCreator::instance().createItem("Inputs", inNode, {0,0});

           Msnhnet::ResBlockLayer *layer = reinterpret_cast<Msnhnet::ResBlockLayer*>(builder.net->layers[clickedIndex-1]);

           node1->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel));

           subScene->addNode(node1);

           std::vector<Msnhnet::BaseLayer *>  baseLayers   = layer->baseLayers;

           qreal widthMax    =   0;

           qreal tmp     =   (baseLayers.size()+1)*256  ;
            if(widthMax < tmp)
            {
                widthMax = tmp;
            }

           Node* node2 = NodeCreator::instance().createItem("AddOutputs", outNode, {widthMax,0});

           node2->attributes()[0]->setData(QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel));
            subScene->addNode(node2);

           subScene->connectNode(inNode,"output",outNode,"input");

           qreal width       =   256;
            qreal height      =   400;

           for (int j = 0; j < baseLayers.size(); ++j)
            {
                QString layerName = QString::fromStdString( baseLayers[j]->layerName).trimmed();

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
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString filters     = QString("%1").arg(layer->outChannel);
        QString kernel      = QString("%1*%2").arg(layer->kSizeX).arg(layer->kSizeY);
        QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
        QString padding     = QString("%1*%2").arg(layer->paddingX).arg(layer->paddingY);
        QString dilation    = QString("%1*%2").arg(layer->dilationX).arg(layer->dilationY);
        QString group       = QString("%1/%2").arg(layer->groupIndex).arg(layer->groups);
        QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input     );
        node->attributes()[1]->setData(filters   );
        node->attributes()[2]->setData(kernel    );
        node->attributes()[3]->setData(stride    );
        node->attributes()[4]->setData(padding   );
        node->attributes()[5]->setData(dilation  );
        node->attributes()[6]->setData(group     );
        node->attributes()[7]->setData(activation);
        node->attributes()[8]->setData(output    );
    }

   if(layerName == "Connected")
    {
        Msnhnet::ConnectedLayer *layer = reinterpret_cast<Msnhnet::ConnectedLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input     );
        node->attributes()[1]->setData(activation);
        node->attributes()[2]->setData(output    );
    }

   if(layerName == "MaxPool")
    {
        Msnhnet::MaxPoolLayer *layer = reinterpret_cast<Msnhnet::MaxPoolLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString filters     = QString("%1").arg(layer->outChannel);
        QString kernel      = QString("%1*%2").arg(layer->kSizeX).arg(layer->kSizeY);
        QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
        QString padding     = QString("%1*%2").arg(layer->paddingX).arg(layer->paddingY);
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(filters);
        node->attributes()[2]->setData(kernel );
        node->attributes()[3]->setData(stride );
        node->attributes()[4]->setData(padding);
        node->attributes()[5]->setData(output );
    }

   if(layerName == "LocalAvgPool")
    {
        Msnhnet::LocalAvgPoolLayer *layer = reinterpret_cast<Msnhnet::LocalAvgPoolLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString filters     = QString("%1").arg(layer->outChannel);
        QString kernel      = QString("%1*%2").arg(layer->kSizeX).arg(layer->kSizeY);
        QString stride      = QString("%1*%2").arg(layer->strideX).arg(layer->strideY);
        QString padding     = QString("%1*%2").arg(layer->paddingX).arg(layer->paddingY);
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input  );
        node->attributes()[1]->setData(filters);
        node->attributes()[2]->setData(kernel );
        node->attributes()[3]->setData(stride );
        node->attributes()[4]->setData(padding);
        node->attributes()[5]->setData(output );
    }

   if(layerName == "Empty")
    {
        Msnhnet::EmptyLayer *layer = reinterpret_cast<Msnhnet::EmptyLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input );
        node->attributes()[1]->setData(output);
    }

   if(layerName == "BatchNorm")
    {
        Msnhnet::BatchNormLayer *layer = reinterpret_cast<Msnhnet::BatchNormLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString activation  = QString::fromStdString(Msnhnet::Activations::getActivationStr(layer->activation));
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input     );
        node->attributes()[1]->setData(activation);
        node->attributes()[2]->setData(output    );
    }

   if(layerName == "Padding")
    {
        Msnhnet::PaddingLayer *layer = reinterpret_cast<Msnhnet::PaddingLayer *>(inLayer);
        QString input       = QString("%1*%2*%3").arg(layer->width).arg(layer->height).arg(layer->channel);
        QString padding     = QString("%1/%2/%3/%4").arg(layer->top).arg(layer->down).arg(layer->left).arg(layer->right);
        QString output      = QString("%1*%2*%3").arg(layer->outWidth).arg(layer->outHeight).arg(layer->outChannel);

       node->attributes()[0]->setData(input);
        node->attributes()[1]->setData(padding);
        node->attributes()[2]->setData(output);
    }
}

void MainWindow::on_actionPowerOff_triggered()
{
    qApp->quit();
}
