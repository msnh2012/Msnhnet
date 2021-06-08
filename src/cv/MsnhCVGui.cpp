#ifdef USE_MSNHCV_GUI
#include "Msnhnet/cv/MsnhCVGui.h"

#include "../3rdparty/stb/stb_image.h"
#include "Msnhnet/cv/MsnhCVMatOp.h"

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>           

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>            

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>          

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
#include <glad/gl.h>            

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
#define GLFW_INCLUDE_NONE       

#include <glbinding/Binding.h>  

#include <glbinding/gl/gl.h>
using namespace gl;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
#define GLFW_INCLUDE_NONE       

#include <glbinding/glbinding.h>

#include <glbinding/gl/gl.h>
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

namespace Msnhnet
{

#ifndef _WIN32
int _kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}
#endif

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void GLErrorMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam)
{
    (void)source;
    (void)id;
    (void)length;
    (void)userParam;
    std::cerr <<
                 "GL CALLBACK:" <<
                 ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "") <<
                 " type:" << type << " severity:" << severity << " message:" <<  message << "\n";
}

std::mutex Gui::mutex;
bool Gui::isRunning = false;
bool Gui::started   = false;
std::thread Gui::th;
std::map<std::string,Mat> Gui::mats;
std::map<std::string,bool> Gui::matInited;
std::map<std::string,unsigned int> Gui::matTextures;
std::map< std::string, std::map< std::string, Plot > > Gui::xyDatas;
std::string Gui::fontPath = "";
float Gui::fontSize = 12.f;

#ifdef _WIN32
BOOL exitWinGui( DWORD fdwCtrlType )
{
    if(fdwCtrlType == CTRL_CLOSE_EVENT)
    {
        Gui::stopIt();
    }
    return true;
}
#else
void exitUnixGui(int signo)
{
    (void)signo;
    Gui::stopIt();
}
#endif

void Gui::startIt()
{
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)exitWinGui,true);
#else
    signal(SIGINT, exitUnixGui);
#endif
    Gui::isRunning = true;
    Gui::th = std::thread(&Gui::run);
}

void Gui::imShow(const std::string &title, Mat &mat)
{
    std::string tmpTitle = title + "_img";
    if(!Gui::started)
    {
        Gui::startIt();
        Gui::started = true;
    }
    Mat tmpMat;
    mat.convertTo(tmpMat,CVT_DATA_TO_U8);
    if(tmpMat.getChannel()==1)
    {
        MatOp::cvtColor(tmpMat,tmpMat,CVT_GRAY2RGBA);
    }
    else if(tmpMat.getChannel()==3)
    {
        MatOp::cvtColor(tmpMat,tmpMat,CVT_RGB2RGBA);
    }

    mutex.lock();

    mats[tmpTitle] = tmpMat;
    matInited[tmpTitle] = false;
    matTextures[tmpTitle] = -1;
    mutex.unlock();
}

void Gui::plotXYData(const std::string &title, const std::string &plotName, const Plot &data, const std::string &xLabel, const std::string &yLabel)
{
    if(!Gui::started)
    {
        Gui::startIt();
        Gui::started = true;
    }

    std::string tmpTitle = title + "_plot"+u8"¤"+xLabel+u8"¤"+yLabel;
    mutex.lock();
    xyDatas[tmpTitle][plotName] = data;
    mutex.unlock();
}

void Gui::plotLine(const std::string &title, const std::string &lineName, const std::vector<Vec2F32> &data)
{
    Plot p1;
    p1.plotType = PLOT_LINE;
    p1.marker   = ImPlotMarker_Circle;
    p1.withPoint = true;
    p1.data     = data;
    plotXYData(title, lineName, p1);
}

void Gui::plotPoints(const std::string &title, const std::string &pointsName, const std::vector<Vec2F32> &data)
{
    Plot p2;
    p2.plotType = PLOT_POINTS;
    p2.marker   = ImPlotMarker_Circle;
    p2.data     = data;
    plotXYData(title, pointsName, p2);
}

void Gui::stopIt()
{
    isRunning = false;
    Gui::th.join();
}

#ifdef linux
int getch(void)
{
    struct termios oldattr, newattr;
    int ch;
    tcgetattr( STDIN_FILENO, &oldattr );
    newattr = oldattr;
    newattr.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newattr );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldattr );
    return ch;
}
#endif

void Gui::wait(int i)
{
    if(i==0)
    {
        int ch;
        while (1)
        {
            if (_kbhit())
            {

                ch = getch();

                if (ch == 27){ break; }

            }
        }
        Gui::stopIt();
    }
    else
    {
#ifdef _WIN32
        _sleep(i);
#else
        sleep(i);
#endif
    }
}

bool Gui::waitEnterKey()
{
    if (_kbhit())
    {

        int ch = getchar();

        if (ch == 10)
        {
            return true;
        }
        else
        {
            return false;
        }

    }
    return false;
}

void Gui::setFont(const std::string &fontPath, const float &size)
{
    Gui::fontPath = fontPath;
    Gui::fontSize = size;
}

void Gui::run()
{

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return;

#ifdef __APPLE__

    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            

#else

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "MsnhCV GUI (Press esc in terminal to exit!)", NULL, NULL);
    if (window == NULL)
        return;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); 

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
    bool err = gladLoadGL(glfwGetProcAddress) == 0; 

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
    bool err = false;
    glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
    bool err = false;
    glbinding::initialize([](const char* name) { return (glbinding::ProcAddress)glfwGetProcAddress(name); });
#else
    bool err = false; 

#endif
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;

    if(fontPath!="")
        io.Fonts->AddFontFromFileTTF(fontPath.c_str(), fontSize, NULL, io.Fonts->GetGlyphRangesChineseFull());

    ImGui::StyleColorsClassic();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    while(isRunning)
    {

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        mutex.lock();

        for (auto &line : xyDatas)
        {
            std::string tmp = line.first;
            std::vector<std::string> listStr;
            ExString::split(listStr, tmp, u8"¤");

            ImGui::Begin(listStr[0].c_str());
            ImGui::SetWindowSize(ImVec2(600, 350));

            std::map< std::string, Plot > lineTmp = line.second;

            if (ImPlot::BeginPlot(listStr[0].c_str(),listStr[1].c_str(),listStr[2].c_str()))
            {
                ImPlot::GetStyle().AntiAliasedLines = true;
                for(auto &l : lineTmp)
                {
                    Plot p = l.second;
                    std::vector<Vec2F32> data = p.data;

                    std::vector<float> x;
                    std::vector<float> y;

                    for (int i = 0; i < data.size(); ++i)
                    {
                        x.push_back(data[i].x1);
                        y.push_back(data[i].x2);
                    }

                    if(p.plotType == PLOT_LINE)
                    {
                        if(p.withPoint)
                        {
                            ImPlot::SetNextMarkerStyle(p.marker);
                        }

                        ImPlot::PlotLine(l.first.c_str(), x.data(), y.data(), x.size());
                    }
                    else if(p.plotType == PLOT_POINTS)
                    {
                        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
                        ImPlot::SetNextMarkerStyle(p.marker, 2, ImVec4(0,1,0,0.5f), IMPLOT_AUTO, ImVec4(0,1,0,1));
                        ImPlot::PlotScatter(l.first.c_str(), x.data(), y.data(), x.size());
                    }

                }
                ImPlot::EndPlot();
            }
            ImGui::End();
        }

        for (auto &init : matInited)
        {
            if(!init.second)
            {
                GLuint texture;
                glGenTextures(1,&texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                Mat tmpMat = mats[init.first];

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,tmpMat.getWidth(), tmpMat.getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmpMat.getData().u8);
                matTextures[init.first] = texture;
                init.second = true;
            }

            if(init.second)
            {
                int width  = mats[init.first].getWidth();
                int height = mats[init.first].getHeight();

                ImGui::Begin(init.first.c_str());   

                ImGui::SetWindowSize(ImVec2(width+20, height+50));
                ImGui::Image((void*)(intptr_t)matTextures[init.first], ImVec2(width, height));
                ImGui::End();
            }
        }

        mutex.unlock();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

}
#endif
