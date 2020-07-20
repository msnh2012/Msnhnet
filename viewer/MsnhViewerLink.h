#ifndef PIPER_LINK_H
#define PIPER_LINK_H

#include "MsnhViewerAttribute.h"
#include <QGraphicsPathItem>

namespace MsnhViewer
{
    class Link : public QGraphicsPathItem
    {
    public:
        Link();
        ~Link();

        void connectFrom(Attribute* from);
        void connectTo(Attribute* to);
        void disconnect();
        bool isConnected();

        void updatePath();
        void updatePath(QPointF const& end);

        void setColor(QColor const& color);

        Attribute const* from() const { return from_; }
        Attribute const* to() const   { return to_;   }

    private:

        void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;

        void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
        void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
        void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

        void updatePath(QPointF const& start, QPointF const& end);

        void computeControlPoint(QPointF const& p0, QPointF const& p1, QPointF const& p2, double t,
                                 QPointF& ctrl1, QPointF& ctrl2);
        void drawSplines(QVector<QPointF> const& waypoints, double t);

        QPen pen_;
        QPen selected_;

        Attribute* from_{nullptr};
        Attribute* to_{nullptr};
    };
}

#endif
