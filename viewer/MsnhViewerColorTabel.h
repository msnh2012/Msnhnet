#ifndef MSNHVIEWERCOLORTABEL_H
#define MSNHVIEWERCOLORTABEL_H
#include <QHash>
#include <QColor>
#include <QDebug>
namespace MsnhViewer
{
class ColorTabel
{
public:
    static ColorTabel &instance();
    void addColor(const QString &key, const QColor &color);
    QColor getColor(const QString &key);
private:
    QHash<QString, QColor> colors;
};
}

#endif 

