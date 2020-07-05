#ifndef MSNHBASELAYER_H
#define MSNHBASELAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/core/MsnhSimd.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class NetworkState;
class MsnhNet_API BaseLayer
{
public:
    BaseLayer();
    virtual ~BaseLayer();

   static bool     supportAvx;
    static bool     supportFma;
    static bool     isPreviewMode;

   LayerType       type;                       

   ActivationType  activation;                 

   std::vector<float> actParams;

   int             num             =  0;       

   size_t          workSpaceSize   =  0;

   int             height          =  0;
    int             width           =  0;
    int             channel         =  0;

   int             outHeight       =  0;
    int             outWidth        =  0;
    int             outChannel      =  0;

   int             inputNum        =  0;
    int             outputNum       =  0;

   size_t          numWeights      =  0;       

   int             batch           =  0;
    float          *output          =  nullptr; 

   float           bFlops          =  0;

   std::string     layerName       =  "BaseLayer";
    std::string     layerDetail     =  "";

   float           forwardTime     =  0;

   static void setPreviewMode(const bool &isPreviewMode);

   virtual void forward(NetworkState &netState);
    virtual void loadAllWeigths(std::vector<float> &weights);

   static void initSimd();
    inline void releaseArr(void * value)
    {
        if(value!=nullptr)
        {
            delete[] value;
            value = nullptr;
        }
    }
};
}

#endif 

