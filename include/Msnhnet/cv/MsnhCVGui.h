#ifndef MSNHCVGUI_H
#define MSNHCVGUI_H
#include <Msnhnet/config/MsnhnetCfg.h>
#ifdef USE_MSNHCV_GUI
#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVThread.h"
#include "Msnhnet/3rdparty/imgui/imgui.h"
#include "Msnhnet/3rdparty/imgui/imgui_impl_glfw.h"
#include "Msnhnet/3rdparty/imgui/imgui_impl_opengl3.h"
#include "Msnhnet/3rdparty/imgui/imgui_memory_editor.h"
#include "Msnhnet/3rdparty/imgui/implot.h"
#include <thread>
#include <map>
#include <mutex>

#ifdef _WIN32
#include <conio.h>
#include <Windows.h>
#else
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif
namespace Msnhnet
{

enum PlotType
{
    PLOT_LINE = 0,
    PLOT_POINTS
};

struct Plot
{
    Plot(){}
    bool withPoint = false;
    PlotType plotType;
    ImPlotMarker_ marker;
    std::vector<Vec2F32> data;
};

class MsnhNet_API Gui
{
public:
    Gui() {}
    static void imShow(const std::string &title, Mat &mat);
    static void plotXYData(const std::string &title, const std::string &plotName, const Plot &data, const std::string &xLabel="X", const std::string &yLabel="Y");
    static void plotLine(const std::string &title, const std::string &lineName, const std::vector<Vec2F32>& data);
    static void plotPoints(const std::string &title, const std::string &pointsName, const std::vector<Vec2F32>& data);
    static void stopIt();
    static void wait(int i=0);
    static bool waitEnterKey();
    static void setFont(const std::string& fontPath, const float &size = 16);
private:

    static void startIt();
    static void run();
    static std::thread th;

    static std::map<std::string,Mat> mats;
    static std::map<std::string,bool> matInited;
    static std::map<std::string,unsigned int> matTextures;

    static std::map< std::string, std::map< std::string, Plot > > xyDatas;

    static bool isRunning;
    static bool started;
    static std::mutex mutex;
    static std::string fontPath;
    static float fontSize;
};
}
#endif
#endif 

