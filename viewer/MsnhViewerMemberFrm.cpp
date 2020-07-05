#include "MsnhViewerMemberFrm.h"

namespace MsnhViewer
{
    MemberForm::MemberForm(QGraphicsItem* parent, QVariant& data, QRectF const& boundingRect, QBrush const& brush)
        : QGraphicsProxyWidget(parent)
        , data_{data}
        , boundingRect_{boundingRect}
        , brush_{brush}
    {

    }

    void MemberForm::paint(QPainter* painter, QStyleOptionGraphicsItem const* option, QWidget* widget)
    {
        painter->setPen(Qt::NoPen);
        painter->setBrush(brush_);
        painter->drawRoundedRect(boundingRect_, 6, 6);

        QGraphicsProxyWidget::paint(painter, option, widget);
    }

    void MemberForm::onDataUpdated(QVariant const& data)
    {
        data_ = data;
    }

}
