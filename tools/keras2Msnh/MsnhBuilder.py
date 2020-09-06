class Msnhnet:
    def __init__(self):
        self.net = ""
        self.index = 0
        self.name_index_dict = dict()

    def buildConfig(self,shape):
        self.net = self.net + "config:\n"
        self.net = self.net + "  batch: " + str(int(shape[0])) + "\n"
        self.net = self.net + "  channels: " + str(int(shape[1])) + "\n"
        self.net = self.net + "  width: " + str(int(shape[2])) + "\n"
        self.net = self.net + "  height: " + str(int(shape[3])) + "\n"
 
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

    def buildActivation(self, name, activation):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "act:\n"
        
        # todo 
        if activation == "relu":
            self.net = self.net + "  activation: relu\n"
        if activation == "relu6":
            self.net = self.net + "  activation: relu6\n"    
        if activation == "sigmoid":
            self.net = self.net + "  activation: logistic\n" 
        if activation == "leakylu":
            self.net = self.net + "  activation: leaky\n" 
        if activation == "linear":
            self.net = self.net + "  activation: none\n" 

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
        
    def buildPooling(self, name, type, kSizeX, kSizeY, strideX, strideY, paddingX, paddingY):
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
        
    def buildConnect(self, name, output):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "connect:\n"
        self.net = self.net + "  output: " + str(int(output)) + "\n"

    def buildUpsample2D(self, name, stride):
        self.name_index_dict[name]=self.index
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "upsample:\n"
        self.net = self.net + "  stride: " + str(int(stride)) + "\n"

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