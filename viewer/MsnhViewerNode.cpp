#include <QGraphicsScene>
#include <QTextDocument>
#include <QGraphicsSceneContextMenuEvent>
#include <QMenu>
#include <QDebug>

#include "MsnhViewerNode.h"
#include "MsnhViewerLink.h"
#include "MsnhViewerMemberFrm.h"
#include "MsnhViewerThemeManager.h"

namespace MsnhViewer
{
    constexpr int attributeHeight = 26;
    constexpr int baseHeight = 26;
    constexpr int baseWidth  = 188;

    NodeName::NodeName(QGraphicsItem* parent) : QGraphicsTextItem(parent)
    {
        QTextOption options;
        options.setWrapMode(QTextOption::NoWrap);
        this->setEnabled(false);
        document()->setDefaultTextOption(options);
    }

    void NodeName::adjustPosition()
    {
        setPos(-(boundingRect().width() - parentItem()->boundingRect().width()) * 0.5, -boundingRect().height());
    }

    void NodeName::keyPressEvent(QKeyEvent* e)
    {
    }

    Node::Node(QString const& type, QString const& name)
        : QGraphicsItem(nullptr)
        , boundingRect_{0, 0, baseWidth, baseHeight}
        , name_{new NodeName(this)}
        , type_{type}
        , mode_{Mode::enable}
        , width_{baseWidth}
        , height_{baseHeight}
        , attributes_{}
    {

        setFlag(QGraphicsItem::ItemIsMovable);
        setFlag(QGraphicsItem::ItemIsSelectable);
        setFlag(QGraphicsItem::ItemIsFocusable);

        name_->setTextInteractionFlags(Qt::TextEditorInteraction);
        setName(name);

        createStyle();

        typeRect_ = QRectF{1, 13, width_ - 2.0, (qreal)attributeHeight};
        height_ += attributeHeight; 
    }

    Node::~Node()
    {
        Scene* pScene = static_cast<Scene*>(scene());
        pScene->removeNode(this);
    }

    void Node::highlight(Attribute* emitter)
    {
        for (auto& attr : attributes_)
        {
            if (attr == emitter)
            {

                continue;
            }

            if (attr->accept(emitter))
            {
                attr->setMode(DisplayMode::highlight);
            }
            else
            {
                attr->setMode(DisplayMode::minimize);
            }
            attr->update();
        }
    }

    void Node::unhighlight()
    {
        for (auto& attr : attributes_)
        {
            attr->setMode(DisplayMode::normal);
            attr->update();
        }
    }

    void Node::createAttributes(QVector<AttributeInfo> const& attributesInfo)
    {
        if (!attributes_.empty())
        {
            qWarning() << "Creating attributes in multiples call is not supported.";
            return;
        }

        QFont attributeFont = ThemeManager::instance().getAttributeTheme().normal.font;
        QFontMetrics metrics(attributeFont);
        QRect boundingRect{0, 0, baseWidth - 32, attributeHeight}; 
        for (auto const& info : attributesInfo)
        {
            boundingRect = boundingRect.united(metrics.boundingRect(info.name));
        }

        boundingRect.setTopLeft({0, 0});
        boundingRect.setWidth(boundingRect.width() + 30); 
        boundingRect.setHeight(attributeHeight);

        width_ = boundingRect.width() + 2;
        typeRect_.setWidth(boundingRect.width());

        for (auto const& info : attributesInfo)
        {
            Attribute* attr{nullptr};
            switch (info.type)
            {
                case AttributeInfo::Type::input:
                {
                    attr = new AttributeInput(this, info, boundingRect);
                    break;
                }
                case AttributeInfo::Type::output:
                {
                    attr = new AttributeOutput(this, info, boundingRect);
                    break;
                }
                case AttributeInfo::Type::member:
                {
                    attr = new AttributeMember(this, info, boundingRect);
                    break;
                }
            }
            attr->setPos(1, 13 + attributeHeight * (attributes_.size() + 1));
            if (attributes_.size() % 2)
            {
                attr->setBackgroundBrush(attributeBrush_);
            }
            else
            {
                attr->setBackgroundBrush(attributeAltBrush_);
            }
            height_ += attributeHeight;
            boundingRect_ = QRectF(0, 0, width_, height_);
            boundingRect_ += QMargins(1, 1, 1, 1);
            attributes_.append(attr);
        }

        prepareGeometryChange();
        name_->adjustPosition(); 
    }

    void Node::createStyle()
    {
        NodeTheme node_theme = ThemeManager::instance().getNodeTheme();
        AttributeTheme attribute_theme = ThemeManager::instance().getAttributeTheme();

        qint32 border = 2;

        backgroundBrush_.setStyle(Qt::SolidPattern);
        backgroundBrush_.setColor(node_theme.background);

        pen_.setStyle(Qt::SolidLine);
        pen_.setWidth(border);
        pen_.setColor(node_theme.border);

        penSelected_.setStyle(Qt::SolidLine);
        penSelected_.setWidth(border);
        penSelected_.setColor(node_theme.border_selected);

        name_->setFont(node_theme.name_font);
        name_->setDefaultTextColor(node_theme.name_color);
        name_->adjustPosition();

        attributeBrush_.setStyle(Qt::SolidPattern);
        attributeBrush_.setColor(attribute_theme.background);
        attributeAltBrush_.setStyle(Qt::SolidPattern);
        attributeAltBrush_.setColor(attribute_theme.background_alt);

        typeBrush_.setStyle(Qt::SolidPattern);
        typeBrush_.setColor(node_theme.type_background);
        typePen_.setStyle(Qt::SolidLine);
        typePen_.setColor(node_theme.type_color);
        typeFont_ = node_theme.type_font;
    }

    QRectF Node::boundingRect() const
    {
        return boundingRect_;
    }

    void Node::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*)
    {

        painter->setBrush(backgroundBrush_);

        if (isSelected())
        {
            painter->setPen(penSelected_);
        }
        else
        {
            painter->setPen(pen_);
        }

        qint32 radius = 10;
        painter->drawRoundedRect(0, 0, width_, height_, radius, radius);

        painter->setBrush(typeBrush_);
        painter->setPen(Qt::NoPen);
        painter->drawRect(typeRect_);

        painter->setFont(typeFont_);
        painter->setPen(typePen_);
        painter->drawText(typeRect_, Qt::AlignCenter, type_);
    }

    void Node::mousePressEvent(QGraphicsSceneMouseEvent* event)
    {

        for (auto& item : scene()->items())
        {
            if (item->zValue() > 1)
            {
                item->setZValue(1);
            }
        }
        setZValue(2);

        QGraphicsItem::mousePressEvent(event);

    }

    void Node::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
    {
        for (auto& attr : attributes_)
        {
            attr->refresh(); 
        }

        QGraphicsItem::mouseMoveEvent(event);
    }

    QString Node::name() const
    {
        return name_->toPlainText();
    }

    void Node::setName(QString const& name)
    {
        name_->setPlainText(name);

        name_->adjustPosition();
    }

    void Node::setMode(Mode mode)
    {
        mode_ = mode;

        for (auto& attribute : attributes_)
        {
            DataTypeTheme theme = ThemeManager::instance().getDataTypeTheme(attribute->dataType());

            if (attribute->isOutput())
            {
                switch (mode)
                {
                    case Mode::enable:  { attribute->setColor(theme.enable);  break; }
                    case Mode::disable: { attribute->setColor(theme.disable); break; }
                    case Mode::neutral: { attribute->setColor(theme.neutral); break; }
                }
            }
        }
    }

    void Node::keyPressEvent(QKeyEvent* event)
    {
        if (isSelected())
        {
            constexpr qreal moveFactor = 5;
            if ((event->key() == Qt::Key::Key_Up)&&(event->modifiers() == Qt::NoModifier))
            {
                moveBy(0, -moveFactor);
            }
            if ((event->key() == Qt::Key::Key_Down)&&(event->modifiers() == Qt::NoModifier))
            {
                moveBy(0, moveFactor);
            }
            if ((event->key() == Qt::Key::Key_Left)&&(event->modifiers() == Qt::NoModifier))
            {
                moveBy(-moveFactor, 0);
            }
            if ((event->key() == Qt::Key::Key_Right)&&(event->modifiers() == Qt::NoModifier))
            {
                moveBy(moveFactor, 0);
            }

            return;
        }

        QGraphicsItem::keyPressEvent(event);
    }

    void Node::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
    {
        if(NodeSelect::selectNode == "null")
            NodeSelect::selectNode = this->name();
         QGraphicsItem::mouseDoubleClickEvent(event);

    }

}
