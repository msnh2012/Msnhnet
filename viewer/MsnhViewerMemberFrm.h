#ifndef PIPER_ATTRIBUTE_MEMBER_H
#define PIPER_ATTRIBUTE_MEMBER_H

#include <QGraphicsProxyWidget>
#include <QPainter>

namespace MsnhViewer
{

    class MemberForm : public QGraphicsProxyWidget
    {
        Q_OBJECT

    public:
        MemberForm(QGraphicsItem* parent, QVariant& data, QRectF const& boundingRect, QBrush const& brush);

    signals:
        void dataUpdated(int);
        void dataUpdated(double);
        void dataUpdated(QString const&);

    public slots:
        void onDataUpdated(QVariant const& data);

    protected:
        void paint(QPainter* painter, QStyleOptionGraphicsItem const*, QWidget*) override;
        QRectF boundingRect() const override { return boundingRect_; }

    private:
        QVariant& data_; 
        QRectF boundingRect_;
        QBrush brush_;
    };

}

#endif
