#ifndef PIPER_ATTRIBUTE_H
#define PIPER_ATTRIBUTE_H

#include <QGraphicsItem>
#include <QPainter>
#include <MsnhViewerMemberFrm.h>

namespace MsnhViewer
{
class Link;

struct AttributeInfo
{
    QString name;
    QString dataType;
    enum Type
    {
        input  = 0,
        output = 1,
        member = 2
    } type;
};
enum DisplayMode
{
    minimize,
    normal,
    highlight
};

class Attribute : public QGraphicsItem
{
public:
    Attribute (QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect);
    ~Attribute();

    AttributeInfo const& info() const { return info_; }
    QString const& name() const     { return info_.name; }
    QString const& dataType() const { return info_.dataType; }
    bool isInput() const  { return (info_.type == AttributeInfo::Type::input);  }
    bool isOutput() const { return (info_.type == AttributeInfo::Type::output); }
    bool isMember() const { return (info_.type == AttributeInfo::Type::member); }

    void setBackgroundBrush(QBrush const& brush) { background_brush_ = brush; }
    virtual void setColor(QColor const& color);

    virtual QPointF connectorPos() const  { return QPointF{}; }
    virtual bool accept(Attribute*) const { return false; }
    void connect(Link* link);
    void disconnect(Link* link) { links_.removeAll(link); }
    void refresh();

    void highlight();

    void unhighlight();

    QVariant const& data() const { return data_; }
    virtual void setData(QVariant const& data) { data_ = data; }

    void setMode(DisplayMode mode)  { mode_ = mode; }

    enum { Type = UserType + 1 };
    int type() const override { return Type; }

protected:
    QRectF boundingRect() const override { return boundingRect_; }
    void paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*) override;

    void applyFontStyle(QPainter* painter, DisplayMode mode);
    void applyStyle(QPainter* painter, DisplayMode mode);

    AttributeInfo info_;
    QVariant data_;
    DisplayMode mode_{DisplayMode::normal};

    QBrush background_brush_;

    QFont minimizeFont_;
    QPen minimizeFontPen_;
    QPen minimizePen_;

    QFont normalFont_;
    QPen normalFontPen_;
    QBrush normalBrush_;
    QPen normalPen_;

    QFont highlightFont_;
    QPen highlightFontPen_;
    QBrush highlightBrush_;
    QPen highlightPen_;

    QRectF boundingRect_;
    QRectF backgroundRect_;
    QRectF labelRect_;
    QVector<Link*> links_;

};

class AttributeMember : public Attribute
{
public:
    AttributeMember(QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect);

    void setData(QVariant const& data) override;
protected:
    QWidget* createWidget();
    MemberForm* form_;

};

class AttributeOutput : public AttributeMember
{
public:
    AttributeOutput(QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect);
    void setColor(QColor const& color) override;
    QPointF connectorPos() const override { return mapToScene(connectorPos_); }

protected:
    void paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

    QRectF connectorRectLeft_;
    QRectF connectorRectRight_;
    QRectF* connectorRect_;
    QPointF connectorPos_;

    Link* newConnection_{nullptr};
};

class AttributeInput : public AttributeMember
{
public:
    AttributeInput(QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect);

    bool accept(Attribute* attribute) const override;
    QPointF connectorPos() const override { return mapToScene(connectorPos_); }

protected:
    void paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

    QPointF inputTriangleLeft_[3];
    QPointF inputTriangleRight_[3];
    QPointF* inputTriangle_;
    QPointF connectorPos_;
};

}

#endif
