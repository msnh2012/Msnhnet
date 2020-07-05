#ifndef PIPER_NODE_CREATOR_H
#define PIPER_NODE_CREATOR_H

#include "MsnhViewerNode.h"
#include "MsnhViewerColorTabel.h"
namespace MsnhViewer
{
    class NodeCreator
    {
    public:
        static NodeCreator& instance();

        QList<QString> availableItems() const { return availableItems_.keys(); }
        void addItem(QString const& type, QVector<AttributeInfo> const& attributes);
        Node* createItem(QString const& type, QString const& name, QPointF const& pos);

    private:
        QHash<QString, QVector<AttributeInfo>> availableItems_;
    };
}

#endif
