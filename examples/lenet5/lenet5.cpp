#include <iostream>
#include "Msnhnet/net/MsnhNetBuilder.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/config/MsnhnetCfg.h"

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: classify /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/lenet5/lenet5.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/lenet5/lenet5.msnhbin";

    std::vector<float>  img{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

	try
	{
		Msnhnet::NetBuilder  msnhNet;
		msnhNet.buildNetFromMsnhNet(msnhnetPath);
		msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

		std::vector<float> result = msnhNet.runClassify(img);

		int bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
		std::cout << "max   : pytorch[19.2447 ]  msnhnet: " << Msnhnet::ExVector::max<float>(result) << std::endl;
                std::cout << "index : pytorch[  5     ]  msnhnet: " << bestIndex << std::endl <<std::flush;
	}
	catch (Msnhnet::Exception &ex)
	{	
		std::cout << ex.what() << " { " << ex.getErrFile() << " " << ex.getErrLine() << "}";
	}
  

    return 0;
}
