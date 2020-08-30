#ifndef PIPER_NODE_H
#define PIPER_NODE_H

#include "MsnhViewerScene.h"
#include "MsnhViewerAttribute.h"
#include "MsnhViewerTypes.h"
#include "MsnhViewerNodeSelect.h"

namespace MsnhViewer
{
    class NodeName : public QGraphicsTextItem
    {
    public:
        NodeName(QGraphicsItem* parent);
        void adjustPosition();

    protected:
        void keyPressEvent(QKeyEvent* e) override;
    };


    class Node : public QGraphicsItem
    {
        friend Link* connect(QString const& from, QString const& out, QString const& to, QString const& in);

    public:
        Node (QString const& type = "", QString const& name = "");
        ~Node();

        // highlight attribute that are compatible with dataType
        void highlight(Attribute* emitter);
        void unhighlight();
        QString name() const;
        QString const& nodeType() const { return type_;  }

        void setMode(Mode mode);
        void setName(QString const& name);
        void setBackgroundColor(QColor const& color)
        {
            backgroundBrush_.setColor(color);
            update();
        }

        // Create attributes of this item.
        void createAttributes(QVector<AttributeInfo> const& attributesInfo);

        QVector<Attribute*> const& attributes() const { return attributes_; }

        QVector<Attribute*>& attributes() { return attributes_; }

    protected:
        QRectF boundingRect() const override;
        void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;

        void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
        void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
        void keyPressEvent(QKeyEvent* event) override;

        void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event);

    private:
        void createStyle();

        QRectF boundingRect_;

        NodeName* name_;
        QString type_;
        QString stage_;
        Mode mode_;

        qint32 width_;
        qint32 height_;

        QBrush backgroundBrush_;
        QPen pen_;
        QPen penSelected_;

        QBrush attributeBrush_;
        QBrush attributeAltBrush_;

        QPen typePen_;
        QBrush typeBrush_;
        QFont typeFont_;
        QRectF typeRect_;

        QVector<Attribute*> attributes_;
    };

    Link* connect(QString const& from, QString const& out, QString const& to, QString const& in);

}

#endif

