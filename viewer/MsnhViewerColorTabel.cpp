#include "MsnhViewerColorTabel.h"

namespace MsnhViewer
{
ColorTabel &ColorTabel::instance()
{
    static ColorTabel colorTabel;
    return colorTabel;
}

void ColorTabel::addColor(const QString &key, const QColor &color)
{
    colors[key] = color;
}

QColor ColorTabel::getColor(const QString &key)
{
    QHash<QString, QColor>::iterator it = colors.find(key);
    if (it == colors.end())
    {
        qWarning() << "Color of node dose not exist , return with black";
        return QColor(0,0,0,255);
    }
    return colors[key];
}

}
