#include "MsnhViewerScene.h"
#include "MsnhViewerNode.h"
#include "MsnhViewerLink.h"
#include "MsnhViewerNodeCreator.h"
#include "MsnhViewerThemeManager.h"

#include <QDebug>

#include <QPainter>
#include <QBrush>
#include <QKeyEvent>
#include <algorithm>
#include <QJsonArray>
#include <QMap>
#include <QMessageBox>
#include <cmath>

namespace MsnhViewer
{
Scene::Scene (QObject* parent)
    : QGraphicsScene(parent)
{

}


Scene::~Scene()
{
    // Manually delete nodes&&links because order are important
    QVector<Node*> deleteNodes = nodes_;
    for (auto& node : deleteNodes)
    {
        delete node;
    }

    QVector<Link*> deleteLinks = links_;
    for (auto& link : deleteLinks)
    {
        delete link;
    }
}

void Scene::drawBackground(QPainter* painter, QRectF const& rect)
{
    QBrush brush(Qt::SolidPattern);
    brush.setColor({50, 50, 50}),
            painter->fillRect(rect, brush);

    QPen pen;
    pen.setColor({100, 100, 100});
    pen.setWidth(2);
    painter->setPen(pen);

    constexpr int gridSize = 20;
    qreal left = int(rect.left()) - (int(rect.left()) % gridSize);
    qreal top = int(rect.top()) - (int(rect.top()) % gridSize);
    QVector<QPointF> points;
    for (qreal x = left; x < rect.right(); x += gridSize)
    {
        for (qreal y = top; y < rect.bottom(); y += gridSize)
        {
            points.append(QPointF(x,y));
        }
    }

    painter->drawPoints(points.data(), points.size());
}


void Scene::keyReleaseEvent(QKeyEvent* keyEvent)
{
    if (keyEvent->key() == Qt::Key::Key_Delete)
    {
        for (auto& item : selectedItems())
        {
            delete item;
        }
    }

    // destroy orphans link
    QVector<Link*> deleteLinks = links_;
    for (auto& link : deleteLinks)
    {
        if (!link->isConnected())
        {
            delete link;
        }
    }
}

qreal Scene::getSceneW() const
{
    return sceneW;
}

void Scene::setSceneW(const qreal &value)
{
    sceneW = value;
}

qreal Scene::getSceneH() const
{
    return sceneH;
}

void Scene::setSceneH(const qreal &value)
{
    sceneH = value>300?value:300;
}



void Scene::addNode(Node* node)
{
    addItem(node);
    nodes_.append(node);
}


void Scene::removeNode(Node* node)
{
    // Remove from mode
    //        for (int i = 0; i < modes_->rowCount(); ++i)
    //        {
    //            QStandardItem* mode = modes_->item(i, 0);
    //            QHash<QString, QVariant> nodeMode = mode->data(Qt::UserRole + 2).toHash();
    //            nodeMode.remove(node->name());
    //            mode->setData(nodeMode, Qt::UserRole + 2);
    //        }

    removeItem(node);
    nodes_.removeAll(node);
}


void Scene::addLink(Link* link)
{
    addItem(link);
    links_.append(link);
}


void Scene::removeLink(Link* link)
{
    removeItem(link);
    links_.removeAll(link);
}


void Scene::connectNode(QString const& from, QString const& out, QString const& to, QString const& in)
{
    auto const nodeFrom = std::find_if(nodes().begin(), nodes().end(),
                                       [&](Node const* node) { return (node->name() == from); }
            );

    auto const nodeTo = std::find_if(nodes().begin(), nodes().end(),
                                     [&](Node const* node) { return (node->name() == to); }
            );

    if (nodeFrom == nodes().end())
    {
        QString error = "Node" + from + "(from) not found";
        linksImportErrors_.append(error);
        return;
    }

    if (nodeTo == nodes().end())
    {
        QString error = "Node" + to + "(to) not found";
        linksImportErrors_.append(error);
        return;
    }

    Attribute* attrOut{nullptr};
    for (auto& attr : (*nodeFrom)->attributes())
    {
        if (attr->isOutput()&&(attr->name() == out))
        {
            attrOut = attr;
            break;
        }
    }

    Attribute* attrIn{nullptr};
    for (auto& attr : (*nodeTo)->attributes())
    {
        if (attr->isInput()&&(attr->name() == in))
        {
            attrIn = attr;
            break;
        }
    }

    if (attrIn == nullptr)
    {
        QString error = "Cannot find attribute" + in + "(in) in the node" + to;
        linksImportErrors_.append(error);
        return;
    }

    if (attrOut == nullptr)
    {
        QString error = "Cannot find attribute" + out + "(out) in the node" + from;
        linksImportErrors_.append(error);
        return;
    }

    if (!attrIn->accept(attrOut))
    {
        QString error = "Cannot connect node" + from + "to node" + to + ". Type mismatch";
        return;
    }

    Link* link = new Link;
    link->connectFrom(attrOut);
    link->connectTo(attrIn);
    addLink(link);
}

void Scene::clear()
{
    while (!links_.isEmpty())
    {
        removeLink(links_[0]);
    }

    while (!nodes_.isEmpty())
    {
        removeNode(nodes_[0]);
    }
}

QColor generateRandomColor()
{
    // procedural color generator: the gold ratio
    static double nextColorHue = 1.0 / (rand() % 100); // don't need a proper random here
    constexpr double golden_ratio_conjugate = 0.618033988749895; // 1 / phi
    nextColorHue += golden_ratio_conjugate;
    nextColorHue = std::fmod(nextColorHue, 1.0);

    QColor nextColor;
    nextColor.setHsvF(nextColorHue, 0.5, 0.99);
    return nextColor;
}
}
