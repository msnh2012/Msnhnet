#include "MsnhViewerLink.h"
#include "MsnhViewerNode.h"
#include "MsnhViewerScene.h"

#include <cmath>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QKeyEvent>
#include <QGraphicsRectItem>

#include <QDebug>

namespace MsnhViewer
{
    Link::Link()
    {
        setFlag(QGraphicsItem::ItemIsSelectable);
        setFlag(QGraphicsItem::ItemIsFocusable);

       pen_.setStyle(Qt::SolidLine);
        pen_.setWidth(2);

       selected_.setStyle(Qt::SolidLine);
        selected_.setColor(QColor(255, 180, 180, 255));
        selected_.setWidth(3);
    }

   Link::~Link()
    {
        disconnect();
        Scene* pScene = static_cast<Scene*>(scene());
        pScene->removeLink(this);
    }

   void Link::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
    {
        if (isSelected())
        {
            setPen(selected_);
        }
        else
        {
            setPen(pen_);
        }

       if (to_ != nullptr)
        {
            updatePath();
        }
        QGraphicsPathItem::paint(painter, option, widget);
    }

   void Link::connectFrom(Attribute* from)
    {
        from_ = from;
        from_->connect(this);
    }

   void Link::connectTo(Attribute* to)
    {
        to_ = to;
        to_->connect(this);
        updatePath();
    }

   void Link::disconnect()
    {
        if (from_ != nullptr)
        {
            from_->disconnect(this);
            from_ = nullptr;
        }

       if (to_ != nullptr)
        {
            to_->disconnect(this);
            to_ = nullptr;
        }
    }

   bool Link::isConnected()
    {
        if ((from_ == nullptr) || (to_ == nullptr))
        {
            return false;
        }
        return true;
    }

   void Link::updatePath()
    {
        updatePath(to_->connectorPos());
    }

   void Link::updatePath(QPointF const& end)
    {
        updatePath(from_->connectorPos(), end);
        setZValue(-1); 

   }

   void Link::setColor(QColor const& color)
    {
        pen_.setColor(color);
    }

   void Link::mousePressEvent(QGraphicsSceneMouseEvent* event)
    {
        Scene* pScene = static_cast<Scene*>(scene());

       setSelected(true);

       to_->disconnect(this);

       updatePath(event->scenePos());

       for (auto& node : pScene->nodes())
        {
            node->highlight(from_);
        }
    }

   void Link::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
    {

       updatePath(event->scenePos());
    }

   void Link::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
    {
        Scene* pScene = static_cast<Scene*>(scene());

       for (auto& node : pScene->nodes())
        {
            node->unhighlight();
        }

       AttributeInput* input = qgraphicsitem_cast<AttributeInput*>(scene()->itemAt(event->scenePos(), QTransform()));
        if (input != nullptr)
        {
            if (input->accept(from_))
            {
                connectTo(input);
            }
        }
        else
        {
            connectTo(to_); 

       }
    }

   void Link::updatePath(QPointF const& start, QPointF const& end)
    {
        qreal dx = (end.x() - start.x()) * 0.5;
        qreal dy = (end.y() - start.y());
        QPointF c1{start.x() + dx, start.y() + dy * 0};
        QPointF c2{start.x() + dx, start.y() + dy * 1};

       QPainterPath path;
        path.moveTo(start);
        path.cubicTo(c1, c2, end);

       setPath(path);
    }

   void Link::computeControlPoint(QPointF const& p0, QPointF const& p1, QPointF const& p2, double t,
                                   QPointF& ctrl1, QPointF& ctrl2)
    {
        using namespace std;

       double d01 = sqrt(pow(p1.x()-p0.x(), 2) + pow(p1.y() - p0.y(), 2));
        double d12 = sqrt(pow(p2.x()-p1.x(), 2) + pow(p2.y() - p1.y(), 2));

       double fa = t * d01 / (d01 + d12);   

       double fb = t * d12 / (d01 + d12);   

       double p1x = p1.x() - fa * (p2.x() - p0.x()); 

       double p1y = p1.y() - fa * (p2.y() - p0.y()); 

       ctrl1.setX(p1x);
        ctrl1.setY(p1y);

       double p2x = p1.x() + fb * (p2.x() - p0.x());
        double p2y = p1.y() + fb * (p2.y() - p0.y());
        ctrl2.setX(p2x);
        ctrl2.setY(p2y);
    }

   void Link::drawSplines(QVector<QPointF> const& waypoints, double t)
    {

       QVector<QPointF> controlPoints;
        for (int i = 0; i < waypoints.size() - 2; i += 1)
        {
            QPointF c1, c2;
            computeControlPoint(waypoints.at(i), waypoints.at(i + 1), waypoints.at(i + 2), t,
                                c1, c2);
            controlPoints << c1 << c2;
        }
        auto nextWaypoint = waypoints.cbegin();
        auto ctrl = controlPoints.cbegin();

       QPainterPath path;
        path.moveTo(*(nextWaypoint++));
        path.quadTo(*(ctrl++), *(nextWaypoint++));

       for (int i = 2; i < waypoints.size() - 1; i += 1)
        {
            path.cubicTo(*ctrl, *(ctrl+1), *(nextWaypoint++));
            ctrl += 2;
        }

       path.quadTo(*ctrl, *nextWaypoint);
        setPath(path);
    }
}
