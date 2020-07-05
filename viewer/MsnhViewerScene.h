#ifndef PIPER_SCENE_H
#define PIPER_SCENE_H

#include <QGraphicsScene>
#include <QStandardItemModel>
#include <QVector>
#include <QJsonObject>

namespace MsnhViewer
{
    class Link;
    class Node;

    class Scene : public QGraphicsScene
    {
        Q_OBJECT

    public:
        Scene(QObject *parent = nullptr);
        ~Scene();

        void addNode(Node* node);
        void removeNode(Node* node);
        QVector<Node*> const& nodes() const { return nodes_; }

        void addLink(Link* link);
        void removeLink(Link* link);
        QVector<Link*> const& links() const { return links_; }
        void connectNode(QString const& from, QString const& out, QString const& to, QString const& in);

        QModelIndex addMode(QString const& name);
        QStandardItemModel* modes()  const { return modes_;  }

        void clear();

        qreal getSceneH() const;
        void setSceneH(const qreal &value);

        qreal getSceneW() const;
        void setSceneW(const qreal &value);

    protected:
        void drawBackground(QPainter *painter, const QRectF &rect) override;
        void keyReleaseEvent(QKeyEvent *keyEvent) override;

    private:
        void placeNodesDefaultPosition();

        QVector<Node*> nodes_;
        QVector<Link*> links_;

        qreal sceneH = 0;
        qreal sceneW = 0;

        QVector<QString> nodesImportErrors_;
        QVector<QString> linksImportErrors_;
        QStandardItemModel* modes_;
    };

    QColor generateRandomColor();
}

#endif

