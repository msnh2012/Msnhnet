#ifndef MSNHCVGUI_H
#define MSNHCVGUI_H

#ifdef USE_MSNHCV_GUI
#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVThread.h"
#include "Msnhnet/3rdparty/imgui/imgui.h"
#include "Msnhnet/3rdparty/imgui/imgui_impl_glfw.h"
#include "Msnhnet/3rdparty/imgui/imgui_impl_opengl3.h"
#include "Msnhnet/3rdparty/imgui/imgui_memory_editor.h"

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
class MsnhNet_API Gui
{
public:
    Gui() {}
    static void imShow(const std::string &title, Mat &mat);
    static void stopIt();
    static void wait(int i=0);
    static bool waitEnterKey();
private:

    static void startIt();
    static void run();
    static std::thread th;
    static std::map<std::string,Mat> mats;
    static std::map<std::string,bool> matInited;
    static std::map<std::string,unsigned int> matTextures;
    static bool isRunning;
    static bool started;
    static std::mutex mutex;
};
}
#endif
#endif 

