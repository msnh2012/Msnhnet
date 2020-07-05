#include "MsnhViewerView.h"
#include <QWheelEvent>
#include <QKeyEvent>
#include <QDebug>

namespace MsnhViewer
{
View::View(QWidget* parent)
    : QGraphicsView(parent)
{
    setFocusPolicy(Qt::ClickFocus);
    setDragMode(QGraphicsView::RubberBandDrag);
    this->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    this->setRenderHints(QPainter::Antialiasing|QPainter::HighQualityAntialiasing|QPainter::NonCosmeticDefaultPen|QPainter::SmoothPixmapTransform|QPainter::TextAntialiasing);
}

void View::fitView()
{
    this->resetMatrix();
    this->scale(1,1);
    this->centerOn(0,0);
}

void View::wheelEvent(QWheelEvent* event)
{
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

   constexpr qreal inFactor = 1.1f;
    constexpr qreal outFactor = 1 / inFactor;

   qreal zoomFactor = outFactor;
    if (event->delta() > 0)
    {
        zoomFactor = inFactor;
    }

   scale(zoomFactor, zoomFactor);
}

void View::keyPressEvent(QKeyEvent* event)
{

   setTransformationAnchor(QGraphicsView::AnchorViewCenter);
    constexpr qreal inFactor = 1.1f;
    constexpr qreal outFactor = 1 / inFactor;

   if ((event->key() == Qt::Key::Key_Plus) && (event->modifiers() & Qt::ControlModifier))
    {
        scale(inFactor, inFactor);
        event->accept();
    }
    if ((event->key() == Qt::Key::Key_Minus) && (event->modifiers() & Qt::ControlModifier))
    {
        scale(outFactor, outFactor);
        event->accept();
    }

   QGraphicsView::keyPressEvent(event);
}

}
