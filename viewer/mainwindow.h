#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <MsnhViewerScene.h>
#include <QPlainTextEdit>
#include <QFileDialog>
#include <Msnhnet/net/MsnhNetBuilder.h>
#include <QSvgGenerator>
#include <omp.h>
#include <QTimer>
#include <QProgressBar>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    static QPlainTextEdit *logger;

private slots:
    //void on_pushButton_clicked();

    void on_actionOpen_triggered();

    void on_actionPrint_triggered();

    void on_actionHand_triggered();

    void doTimer();

    void startProgressBar();

    void stopProgressBar();

    void updateProgressBar(const int &val);

    void createNode(MsnhViewer::Node *node, const QString &layerName, Msnhnet::BaseLayer *inLayer);

    void on_actionPowerOff_triggered();

private:
    Ui::MainWindow *ui;
    MsnhViewer::Scene *scene;
    MsnhViewer::Scene *subScene;
    QTimer *timer;
    Msnhnet::NetBuilder builder;
    QProgressBar *progressBar;
};
#endif // MAINWINDOW_H
