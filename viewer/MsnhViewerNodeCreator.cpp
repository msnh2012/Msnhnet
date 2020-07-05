#include "MsnhViewerNodeCreator.h"
#include <QDebug>

namespace MsnhViewer
{
    NodeCreator& NodeCreator::instance()
    {
        static NodeCreator creator_;
        return creator_;
    }

    void NodeCreator::addItem(QString const& type, QVector<AttributeInfo> const& attributes)
    {
        QHash<QString, QVector<AttributeInfo>>::iterator it = availableItems_.find(type);
        if (it != availableItems_.end())
        {
            qDebug() << "Can't add the item. Type" << type << "already exists.";
            return;
        }
        availableItems_.insert(type, attributes);
    }

    Node* NodeCreator::createItem(QString const& type, QString const& name, const QPointF& pos)
    {
        QHash<QString, QVector<AttributeInfo>>::iterator it = availableItems_.find(type);
        if (it == availableItems_.end())
        {
            qDebug() << "Can't create the item" << name << ". Type" << type << "is unknown";
            return nullptr;
        }

        Node* node = new Node(type, name);
        node->setPos(pos);
        node->setBackgroundColor(ColorTabel::instance().getColor(type));
        node->createAttributes(*it);
        return node;
    }
}
