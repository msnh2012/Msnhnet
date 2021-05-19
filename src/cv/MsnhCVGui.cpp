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

    mats[title] = tmpMat;
    matInited[title] = false;
    matTextures[title] = -1;
    mutex.unlock();
}

void Gui::stopIt()
{
    isRunning = false;
    Gui::th.join();
}

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

    GLFWwindow* window = glfwCreateWindow(1280, 720, "MsnhCV GUI", NULL, NULL);
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
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    while(isRunning)
    {

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        mutex.lock();
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
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

}
#endif
