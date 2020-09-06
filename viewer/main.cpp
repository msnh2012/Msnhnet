#include "mainwindow.h"
#include <QDateTime>
#include <QApplication>
#include <MsnhViewerColorTabel.h>
#include <MsnhViewerNodeCreator.h>
#include <MsnhViewerThemeManager.h>
using namespace MsnhViewer;

void messageHandler(QtMsgType type, const QMessageLogContext& context,
                    const QString &msg)
{

    QString htmlInfoHead=("<p style='color: rgb(0, 220, 0)'>");
    QString htmlCriticalHead=("<p style='color: rgb(220, 80, 0)'>");
    QString htmlFatalHead=("<p style='color: rgb(255, 0, 0)'>");
    QString htmlWarningHead = ("<p style='color: rgb(255, 191, 36)'>");
    QString htmlEnd = ("</p>");
    QString str;
    switch (type) {
    case QtDebugMsg:    str.append(htmlInfoHead + QDateTime::currentDateTime().toString("yyyy/MM.dd HH:mm:ss") + " DBG"); break;
    case QtInfoMsg:     str.append(htmlInfoHead + QDateTime::currentDateTime().toString("yyyy/MM.dd HH:mm:ss") + " INF"); break;
    case QtWarningMsg:  str.append(htmlWarningHead + QDateTime::currentDateTime().toString("yyyy/MM.dd HH:mm:ss") + " WRN"); break;
    case QtCriticalMsg: str.append(htmlCriticalHead + QDateTime::currentDateTime().toString("yyyy/MM.dd HH:mm:ss") + " CRT"); break;
    case QtFatalMsg:    str.append(htmlFatalHead + QDateTime::currentDateTime().toString("yyyy/MM.dd HH:mm:ss") + " FTL"); break;
    }

    str.append(" (" + QString(context.category) + "): " + msg + htmlEnd);

    MainWindow::logger->appendHtml(str);
}
int main(int argc, char *argv[])
{

    if (!ThemeManager::instance().load(":/theme/theme.json"))
    {
        return 1;
    }
    qInstallMessageHandler(messageHandler);
    QApplication a(argc, argv);
    MainWindow w;
    w.showMaximized();
    return a.exec();
}

