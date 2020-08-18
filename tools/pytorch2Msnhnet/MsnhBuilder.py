from collections import OrderedDict
import sys

class Msnhnet:
    def __init__(self):
        self.inAddr = ""
        self.net = ""
        self.index = 0
        self.name_index_dict = OrderedDict()

    def getLastKey(self):
        return next(reversed(self.name_index_dict))

    def checkInput(self, inAddr,fun):

        if self.index == 0:
            return

        if str(inAddr._cdata) != self.getLastKey():
            try:
                ID = self.name_index_dict[str(inAddr._cdata)]
                self.buildRoute(str(inAddr._cdata),str(ID),False)
            except:
                raise NotImplementedError("last op is not supported " + fun + str(inAddr._cdata))
            

    def buildConfig(self, inAddr, shape):
        self.inAddr = inAddr
        self.net = self.net + "config:\n"
        self.net = self.net + "  batch: " + str(int(shape[0])) + "\n"
        self.net = self.net + "  channels: " + str(int(shape[1])) + "\n"
        self.net = self.net + "  height: " + str(int(shape[2])) + "\n"
        self.net = self.net + "  width: " + str(int(shape[3])) + "\n"

 
    def buildConv2d(self, name, filters, kSizeX, kSizeY, paddingX, paddingY, strideX, strideY, dilationX, dilationY, groups, useBias):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "conv:\n"
        self.net = self.net + "  filters: " + str(int(filters)) + "\n"
        self.net = self.net + "  kSizeX: " + str(int(kSizeX)) + "\n"
        self.net = self.net + "  kSizeY: " + str(int(kSizeY)) + "\n"
        self.net = self.net + "  paddingX: " + str(int(paddingX)) + "\n"
        self.net = self.net + "  paddingY: " + str(int(paddingY)) + "\n"
        self.net = self.net + "  strideX: " + str(int(strideX)) + "\n"
        self.net = self.net + "  strideY: " + str(int(strideY)) + "\n"
        self.net = self.net + "  dilationX: " + str(int(dilationX)) + "\n"
        self.net = self.net + "  dilationY: " + str(int(dilationY)) + "\n"
        self.net = self.net + "  groups: " + str(int(groups)) + "\n"
        self.net = self.net + "  useBias: " + str(int(useBias)) + "\n"

    def buildActivation(self, name, activation, params=None):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "act:\n"
        
        # todo 
        if activation == "selu":
            self.net = self.net + "  activation: selu\n"
            return
        if activation == "elu":
            self.net = self.net + "  activation: elu\n"
            return
        if activation == "relu":
            self.net = self.net + "  activation: relu\n"
            return
        if activation == "relu6":
            self.net = self.net + "  activation: relu6\n"    
            return
        if activation == "sigmoid":
            self.net = self.net + "  activation: logistic\n" 
            return
        if activation == "leaky":
            self.net = self.net + "  activation: leaky,"+str(params)+"\n"
            return
        if activation == "tanh":
            self.net = self.net + "  activation: tanh\n" 
            return
        if activation == "logistic":
            self.net = self.net + "  activation: logistic\n" 
            return
        if activation == "softplus":
            self.net = self.net + "  activation: softplus,"+str(params)+"\n"
            return
        if activation == "linear":
            self.net = self.net + "  activation: none\n"
            return
        
        raise NotImplementedError("unknown actiavtion : "+activation)
        

    def buildSoftmax(self, name):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "softmax:\n"
        self.net = self.net + "  groups: 1\n"
    
    def buildBatchNorm(self, name):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "batchnorm:\n  activation: none\n"

    def buildGlobalAvgPooling(self, name):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "globalavgpool:\n  "
        
    def buildPooling(self, name, type, kSizeX, kSizeY, strideX, strideY, paddingX, paddingY, ceilMode):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        if type == "MAX" :
            self.net = self.net + "maxpool:\n"
        else:
            self.net = self.net + "localavgpool:\n"

        self.net = self.net + "  kSizeX: " + str(int(kSizeX)) + "\n"
        self.net = self.net + "  kSizeY: " + str(int(kSizeY)) + "\n"
        self.net = self.net + "  paddingX: " + str(int(paddingX)) + "\n"
        self.net = self.net + "  paddingY: " + str(int(paddingY)) + "\n"
        self.net = self.net + "  strideX: " + str(int(strideX)) + "\n"
        self.net = self.net + "  strideY: " + str(int(strideY)) + "\n"
        self.net = self.net + "  ceilMode: " + str(int(ceilMode)) + "\n"

    def buildConnect(self, name, output, useBias):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "connect:\n"
        self.net = self.net + "  output: " + str(int(output)) + "\n"
        self.net = self.net + "  useBias: " + str(int(useBias)) + "\n"

    def buildUpsample2D(self, name, stride, scale):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "upsample:\n"
        self.net = self.net + "  stride: " + str(int(stride)) + "\n"
        self.net = self.net + "  scale: " + str(float(scale)) + "\n"

    def buildRoute(self, name, layers, addModel):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "route:\n"
        self.net = self.net + "  layers: " + layers + "\n"
        self.net = self.net + "  addModel: " + str(int(addModel)) + "\n"

    def buildPadding(self, name, top, down, left, right):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "padding:\n"
        self.net = self.net + "  top: " + str(int(top)) + "\n"
        self.net = self.net + "  down: " + str(int(down)) + "\n"
        self.net = self.net + "  left: " + str(int(left)) + "\n"
        self.net = self.net + "  right: " + str(int(right)) + "\n"
        self.net = self.net + "  paddingVal: 0\n"
    
    def buildVariableOp(self, name, layers, mode):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "varop:\n"
        self.net = self.net + "  layers: " + layers + "\n"
        self.net = self.net + "  mode: "   + mode + "\n"