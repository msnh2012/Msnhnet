#ifndef PIPER_VIEW_H
#define PIPER_VIEW_H

#include <QGraphicsView>
#include <MsnhViewerScene.h>

namespace MsnhViewer
{
    class View : public QGraphicsView
    {
    public:
        View(QWidget* parent = nullptr);
        void fitView();
    protected:
        void wheelEvent(QWheelEvent* event) override;
        void keyPressEvent(QKeyEvent * event) override;

    };
}

#endif
