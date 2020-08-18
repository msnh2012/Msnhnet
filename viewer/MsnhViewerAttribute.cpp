#include "MsnhViewerAttribute.h"
#include "MsnhViewerLink.h"
#include "MsnhViewerNode.h"
#include "MsnhViewerScene.h"
#include "MsnhViewerThemeManager.h"

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsTextItem>
#include <QMargins>
#include <QDebug>
#include <QLineEdit>
#include <QSpinBox>

#include <type_traits>

namespace MsnhViewer
{

Attribute::Attribute (QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect)
    : QGraphicsItem(parent)
    , info_{info}
    , boundingRect_{boundingRect}
    , backgroundRect_{boundingRect_}
    , labelRect_{boundingRect_.left() + 15, boundingRect_.top(),
                 boundingRect_.width() / 4, boundingRect_.height()}
{
    AttributeTheme theme = ThemeManager::instance().getAttributeTheme();
    DataTypeTheme typeTheme = ThemeManager::instance().getDataTypeTheme(dataType());

    minimizePen_.setStyle(Qt::SolidLine);
    minimizePen_.setWidth(theme.minimize.connector.border_width);
    minimizePen_.setColor(theme.minimize.connector.border_color);
    minimizeFont_ = theme.minimize.font;
    minimizeFontPen_.setStyle(Qt::SolidLine);
    minimizeFontPen_.setColor(theme.minimize.font_color);

    normalPen_.setStyle(Qt::SolidLine);
    normalPen_.setWidth(theme.normal.connector.border_width);
    normalPen_.setColor(theme.normal.connector.border_color);
    normalBrush_.setStyle(Qt::SolidPattern);
    normalBrush_.setColor(typeTheme.enable);
    normalFont_ = theme.normal.font;
    normalFontPen_.setStyle(Qt::SolidLine);
    normalFontPen_.setColor(theme.normal.font_color);

    highlightPen_.setStyle(Qt::SolidLine);
    highlightPen_.setWidth(theme.highlight.connector.border_width);
    highlightPen_.setColor(theme.highlight.connector.border_color);
    highlightBrush_.setStyle(Qt::SolidPattern);
    highlightBrush_.setColor(typeTheme.enable);
    highlightFont_ = theme.highlight.font;
    highlightFontPen_.setStyle(Qt::SolidLine);
    highlightFontPen_.setColor(theme.highlight.font_color);

    prepareGeometryChange();
}


Attribute::~Attribute()
{
    // Disconnect related links.
    for (auto& link : links_)
    {
        link->disconnect();
    }
}


void Attribute::setColor(QColor const& color)
{
    normalBrush_.setColor(color);
    highlightBrush_.setColor(color);
    update();
}


void Attribute::connect(Link* link)
{
    DataTypeTheme typeTheme = ThemeManager::instance().getDataTypeTheme(dataType());
    links_.append(link);
    link->setColor(typeTheme.enable);
}



void Attribute::refresh()
{
    for (auto& link : links_)
    {
        link->updatePath();
    }
}


void Attribute::applyFontStyle(QPainter* painter, DisplayMode mode)
{
    switch (mode)
    {
    case DisplayMode::highlight:
    {
        painter->setFont(highlightFont_);
        painter->setPen(highlightFontPen_);
        break;
    }
    case DisplayMode::normal:
    {
        painter->setFont(normalFont_);
        painter->setPen(normalFontPen_);
        break;
    }
    case DisplayMode::minimize:
    {
        painter->setFont(minimizeFont_);
        painter->setPen(minimizeFontPen_);
        break;
    }
    }
}

void Attribute::applyStyle(QPainter* painter, DisplayMode mode)
{
    switch (mode)
    {
    case DisplayMode::highlight:
    {
        painter->setBrush(highlightBrush_);
        painter->setPen(highlightPen_);
        break;
    }
    case DisplayMode::normal:
    {
        painter->setBrush(normalBrush_);
        painter->setPen(normalPen_);
        break;
    }
    case DisplayMode::minimize:
    {
        painter->setBrush(background_brush_);
        painter->setPen(minimizePen_);
        break;
    }
    }
}


void Attribute::paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*)
{
    // NodeAttribute background.
    painter->setBrush(background_brush_);
    painter->setPen(Qt::NoPen);
    painter->drawRect(backgroundRect_);

    // NodeAttribute label.
    applyFontStyle(painter, mode_);
    painter->drawText(labelRect_, Qt::AlignVCenter, name());
}



AttributeOutput::AttributeOutput(QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect)
    : AttributeMember (parent, info, boundingRect)
{

    // Compute connector rectangle.
    qreal const length = boundingRect_.height() / 4.0;

    connectorRectLeft_  = QRectF{ boundingRect_.left() - length - 1, length,
            length * 2, length * 2 };

    connectorRectRight_ = QRectF{ boundingRect_.right() - length + 1, length,
            length * 2, length * 2 };

    // Update bounding rect to include connector positions
    boundingRect_ = boundingRect_.united(connectorRectLeft_);
    boundingRect_ = boundingRect_.united(connectorRectRight_);
    boundingRect_ += QMargins(20, 0, 20, 0);

    connectorRect_ = &connectorRectRight_;
    // Compute connector center to position the path.
    connectorPos_ = { connectorRect_->x() + connectorRect_->width()  / 2.0,
                      connectorRect_->y() + connectorRect_->height() / 2.0 };
    prepareGeometryChange();
}


void AttributeOutput::setColor(QColor const& color)
{
    Attribute::setColor(color);
    for (auto& link : links_)
    {
        link->setColor(color);
    }
}


void AttributeOutput::paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*)
{
    // Draw generic part (label&&background).
    AttributeMember::paint(painter, nullptr, nullptr);

    applyStyle(painter, mode_);
    painter->drawEllipse(*connectorRect_);
}


void AttributeOutput::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    Scene* pScene = static_cast<Scene*>(scene());

    if (connectorRect_->contains(event->pos())&&event->button() == Qt::LeftButton)
    {
        newConnection_ = new Link();
        newConnection_->connectFrom(this);
        newConnection_->setColor(normalBrush_.color());
        pScene->addLink(newConnection_);

        for (auto const& item : pScene->nodes())
        {
            item->highlight(this);
        }

        return;
    }

    if (event->button() == Qt::MiddleButton)
    {
        setData(!data_.toBool());
    }

    Attribute::mousePressEvent(event);
}


void AttributeOutput::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (newConnection_ == nullptr)
    {
        // Nothing to do
        Attribute::mouseMoveEvent(event);
        return;
    }

    newConnection_->updatePath(event->scenePos());
}


void AttributeOutput::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    Scene* pScene =  static_cast<Scene*>(scene());
    if ((newConnection_ == nullptr) || (event->button() != Qt::LeftButton))
    {
        // Nothing to do
        Attribute::mouseReleaseEvent(event);
        return;
    }

    // Disable highlight
    for (auto const& item : pScene->nodes())
    {
        item->unhighlight();
    }

    AttributeInput* input = qgraphicsitem_cast<AttributeInput*>(scene()->itemAt(event->scenePos(), QTransform()));
    if (input != nullptr)
    {
        if (input->accept(this))
        {
            newConnection_->connectTo(input);
            newConnection_ = nullptr; // connection finished.
            return;
        }
    }

    // cleanup unfinalized connection.
    delete newConnection_;
    newConnection_ = nullptr;
}



AttributeInput::AttributeInput (QGraphicsItem* parent, AttributeInfo const& info, QRect const& boundingRect)
    : AttributeMember (parent, info, boundingRect)
{
    //        data_ = false; // Use data member to store connector position.

    // Compute input inputTriangle_
    qreal length = boundingRect_.height() / 4.0;
    inputTriangleLeft_[0] = QPointF(boundingRect_.left() - 1, length);
    inputTriangleLeft_[1] = QPointF(boundingRect_.left() + length * 1.5, length * 2);
    inputTriangleLeft_[2] = QPointF(boundingRect_.left() - 1, length * 3);

    inputTriangleRight_[0] = QPointF(boundingRect_.right() + 1, length);
    inputTriangleRight_[1] = QPointF(boundingRect_.right() - length * 1.5, length * 2);
    inputTriangleRight_[2] = QPointF(boundingRect_.right() + 1, length * 3);

    boundingRect_ += QMargins(2, 0, 2, 0);


    //   inputTriangle_ = inputTriangleRight_;

    inputTriangle_ = inputTriangleLeft_;

    // Compute connector center to position the path.
    qreal x = inputTriangle_[0].x();;
    qreal y = inputTriangle_[2].y() - inputTriangle_[0].y();
    connectorPos_ = { x, y };
    prepareGeometryChange();
}



bool AttributeInput::accept(Attribute* attribute) const
{
    if (attribute->dataType() != dataType())
    {
        // Incompatible type.
        return false;
    }

    if (attribute->parentItem() == parentItem())
    {
        // can't be connected to another attribute of the same item.
        return false;
    }

    for (auto& link : links_)
    {
        if (link->from() == attribute)
        {
            // We are already connected to this guy.
            return false;
        }
    }

    return true;
}


void AttributeInput::paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*)
{
    // Draw generic part (label&&background).
    AttributeMember::paint(painter, nullptr, nullptr);

    applyStyle(painter, mode_);
    painter->drawConvexPolygon(inputTriangle_, 3);
}


void AttributeInput::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    if (event->button() == Qt::MiddleButton)
    {
        setData(!data_.toBool());
    }
    Attribute::mousePressEvent(event);
}



AttributeMember::AttributeMember(QGraphicsItem* parent, AttributeInfo const& info, const QRect& boundingRect)
    : Attribute (parent, info, boundingRect)
{
    // Reduce the label area to add the form.
    labelRect_ = QRectF{boundingRect_.left() + 15, boundingRect_.top(),
            boundingRect_.width() / 3, boundingRect_.height()};

    // Construct the form (area, background color, widget, widgets options etc).
    QRectF formRect{0, 0, boundingRect_.width() * 2 / 3 - 20, boundingRect_.height() - 10};
    QBrush brush {{180, 180, 180, 255}, Qt::SolidPattern};
    form_ = new MemberForm(this, data_, formRect, brush);

    QWidget* widget = createWidget();
    if (widget != nullptr)
    {
        widget->setFont(normalFont_);
        widget->setMaximumSize(formRect.width(), formRect.height());
        widget->resize(formRect.width(), formRect.height());

        QFile File(":/style.qss");
        File.open(QFile::ReadOnly);
        QString StyleSheet = QLatin1String(File.readAll());
        widget->setStyleSheet(StyleSheet);
        form_->setWidget(widget);
    }
    form_->setPos(labelRect_.right(), labelRect_.top() + 5);
}

void AttributeMember::setData(QVariant const& data)
{
    switch (data.type())
    {
    case QVariant::Type::Int:
    {
        form_->dataUpdated(data.toInt());
        break;
    }
    case QVariant::Type::Double:
    {
        form_->dataUpdated(data.toDouble());
        break;
    }
    case QVariant::Type::String:
    {
        form_->dataUpdated(data.toString());
        break;
    }
    default:
    {
        qDebug() << "Incompatible type: " << data << ". Do nothing";
    }
    }
}

QWidget* AttributeMember::createWidget()
{
    QStringList supportedTypes;

    supportedTypes << "int" << "integer" << "int32_t" << "int64_t";
    if (supportedTypes.contains(dataType()))
    {
        QSpinBox* box = new QSpinBox();
        data_ = box->value();
        box->setMaximum(std::numeric_limits<int>::max());
        box->setMinimum(std::numeric_limits<int>::min());
        QObject::connect(box, QOverload<int>::of(&QSpinBox::valueChanged), form_, &MemberForm::onDataUpdated);
        QObject::connect(form_, SIGNAL(dataUpdated(int)), box, SLOT(setValue(int)));
        return box;
    }

    supportedTypes.clear();
    supportedTypes << "float" << "double" << "real" << "float32_t" << "float64_t";
    if (supportedTypes.contains(dataType()))
    {
        QDoubleSpinBox* box = new QDoubleSpinBox();
        data_ = box->value();
        box->setMaximum(std::numeric_limits<double>::max());
        box->setMinimum(std::numeric_limits<double>::min());
        box->setDecimals(3);
        QObject::connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), form_, &MemberForm::onDataUpdated);
        QObject::connect(form_, SIGNAL(dataUpdated(double)), box, SLOT(setValue(double)));
        return box;
    }

    supportedTypes.clear();
    supportedTypes << "string";
    if (supportedTypes.contains(dataType()))
    {
        QLineEdit* lineEdit = new QLineEdit();
        lineEdit->setAlignment(Qt::AlignCenter);
        lineEdit->setReadOnly(true);
        data_ = lineEdit->text();
        lineEdit->setFont(normalFont_);
        QObject::connect(lineEdit, &QLineEdit::textChanged, form_, &MemberForm::onDataUpdated);
        QObject::connect(form_, SIGNAL(dataUpdated(QString const&)), lineEdit, SLOT(setText(QString const&)));
        return lineEdit;
    }

    return nullptr;
}

}
