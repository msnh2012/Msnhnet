# origin code from https://github.com/UltronAI/pytorch-caffe
# Updated by BBuf
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from collections import OrderedDict
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
from itertools import product as product
from prototxt import *

SUPPORTED_LAYERS = ['Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 
                    'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss', 
                    'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput']

class CaffeData(nn.Module):
    def __init__(self, layer):
        super(CaffeData, self).__init__()
        net_info = OrderedDict()
        props = OrderedDict()
        props['name'] = 'temp network'
        net_info['props'] = props
        print('CaffeData init phase = %s' % (layer['include']['phase']))
        if 'include' in layer:
            layer.pop('include')
        net_info['layers'] = [layer]

        rand_val = random.random()
        protofile = '.temp_data%f.prototxt' % rand_val
        save_prototxt(net_info, protofile)
        weightfile = '.temp_data%f.caffemodel' % rand_val
        open(weightfile, 'w').close()
        caffe.set_mode_cpu()
        if 'include' in layer and layer['include'] == 'TRAIN':
            self.net = caffe.Net(protofile, weightfile, caffe.TRAIN)
        else:
            self.net = caffe.Net(protofile, weightfile, caffe.TEST)
        self.register_buffer('data', torch.zeros(1))
        self.register_buffer('label', torch.zeros(1))
    def __repr__(self):
        return 'CaffeData()'
    def forward(self):
        self.net.forward()
        data = self.net.blobs['data'].data
        label = self.net.blobs['label'].data
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        self.data.resize_(data.size()).copy_(data)
        self.label.resize_(label.size()).copy_(label)
        #print('dataloader data size = %s' % (str(self.data.size())))
        return Variable(self.data), Variable(self.label)

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x

class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):

        if x.dim() == 2:
            nB = x.size(0)
            nC = x.size(1)
            x = x * self.weight.view(1, nC).expand(nB, nC) + \
                self.bias.view(1, nC).expand(nB, nC)
        else:
            nB = x.size(0)
            nC = x.size(1)
            nH = x.size(2)
            nW = x.size(3)
            x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + \
                self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x

class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset
    def __repr__(self):
        return 'Crop(axis=%d, offset=%d)' % (self.axis, self.offset)

    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, Variable(indices))
        return x

class Slice(nn.Module):
   def __init__(self, axis, slice_points):
       super(Slice, self).__init__()
       self.axis = axis
       self.slice_points = slice_points

   def __repr__(self):
        return 'Slice(axis=%d, slice_points=%s)' % (self.axis, self.slice_points)

   def forward(self, x):
       prev = 0
       outputs = []
       is_cuda = x.data.is_cuda
       if is_cuda: device_id = x.data.get_device()
       for idx, slice_point in enumerate(self.slice_points):
           rng = range(prev, slice_point)
           rng = torch.LongTensor(rng)
           if is_cuda: rng = rng.cuda(device_id)
           rng = Variable(rng)
           y = x.index_select(self.axis, rng)
           prev = slice_point
           outputs.append(y)
       return tuple(outputs)

class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Concat(axis=%d)' % self.axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)

class Permute(nn.Module):
    def __init__(self, order0, order1, order2, order3):
        super(Permute, self).__init__()
        self.order0 = order0
        self.order1 = order1
        self.order2 = order2
        self.order3 = order3

    def __repr__(self):
        return 'Permute(%d, %d, %d, %d)' % (self.order0, self.order1, self.order2, self.order3)

    def forward(self, x):
        x = x.permute(self.order0, self.order1, self.order2, self.order3).contiguous()
        return x

class Softmax(nn.Module):
    def __init__(self, axis):
        super(Softmax, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Softmax(axis=%d)' % self.axis

    def forward(self, x):
        assert(self.axis == len(x.size())-1)
        orig_size = x.size()        
        dims = x.size(self.axis)
        x = F.softmax(x.view(-1, dims))
        x = x.view(*orig_size)
        return x

class SoftmaxWithLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super(SoftmaxWithLoss, self).__init__()
    def __repr__(self):
        return 'SoftmaxWithLoss()'
    def forward(self, input, targets):
        targets = targets.long()
        return nn.CrossEntropyLoss.forward(self, input, targets)


class Normalize(nn.Module):
    def __init__(self,n_channels, scale=1.0):
        super(Normalize,self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale
        self.register_parameter('bias', None)

    def __repr__(self):
        return 'Normalize(channels=%d, scale=%f)' % (self.n_channels, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm * self.weight.view(1,-1,1,1)
        return x

class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Flatten(axis=%d)' % self.axis

    def forward(self, x):
        left_size = 1
        for i in range(self.axis):
            left_size = x.size(i) * left_size
        return x.view(left_size, -1).contiguous()


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
    def __repr__(self):
        return 'Accuracy()'
    def forward(self, output, label):
        max_vals, max_ids = output.data.max(1)
        n_correct = (max_ids.view(-1).float() == label.data).sum()
        batchsize = output.data.size(0)
        accuracy = float(n_correct)/batchsize
        print('accuracy = %f', accuracy)
        accuracy = output.data.new().resize_(1).fill_(accuracy)
        return Variable(accuracy)

class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """
    def __init__(self, min_size, max_size, aspects, clip, flip, step, offset, variances):
        super(PriorBox, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.aspects = aspects
        self.clip = clip
        self.flip = flip
        self.step = step
        self.offset = offset
        self.variances = variances

    def __repr__(self):
        return 'PriorBox(min_size=%f, max_size=%f, clip=%d, step=%d, offset=%f, variances=%s)' % (self.min_size, self.max_size, self.clip, self.step, self.offset, self.variances)
        
    def forward(self, feature, image):
        mean = []
        #assert(feature.size(2) == feature.size(3))
        #assert(image.size(2) == image.size(3))
        feature_height = feature.size(2)
        feature_width = feature.size(3)
        image_height = image.size(2)
        image_width = image.size(3)
        #for i, j in product(range(feature_height), repeat=2):
        for j in range(feature_height):
            for i in range(feature_width):
                # unit center x,y
                cx = (i + self.offset) * self.step / image_width
                cy = (j + self.offset) * self.step / image_height
                mw = float(self.min_size)/image_width
                mh = float(self.min_size)/image_height
                mean += [cx-mw/2.0, cy-mh/2.0, cx+mw/2.0, cy+mh/2.0]

                if self.max_size > self.min_size:
                    ww = math.sqrt(mw * float(self.max_size)/image_width)
                    hh = math.sqrt(mh * float(self.max_size)/image_height)
                    mean += [cx-ww/2.0, cy-hh/2.0, cx+ww/2.0, cy+hh/2.0]
                    for aspect in self.aspects:
                        ww = mw * math.sqrt(aspect)
                        hh = mh / math.sqrt(aspect)
                        mean += [cx-ww/2.0, cy-hh/2.0, cx+ww/2.0, cy+hh/2.0]
                        if self.flip:
                            ww = mw / math.sqrt(aspect)
                            hh = mh * math.sqrt(aspect)
                            mean += [cx-ww/2.0, cy-hh/2.0, cx+ww/2.0, cy+hh/2.0]

        # back to torch land
        output1 = torch.Tensor(mean).view(-1, 4)
        output2 = torch.FloatTensor(self.variances).view(1,4).expand_as(output1)
        if self.clip:
            output1.clamp_(max=1, min=0)
        output1 = output1.view(1,1,-1)
        output2 = output2.contiguous().view(1,1,-1)
        output = torch.cat([output1, output2], 1)
        if feature.data.is_cuda:
            device_id = feature.data.get_device()
            return Variable(output.cuda(device_id))
        else:
            return Variable(output)

class CaffeNet(nn.Module):
    def __init__(self, protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TEST'):
        super(CaffeNet, self).__init__()
        self.omit_data_layer = omit_data_layer
        self.phase = phase
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info, width, height, channels)
        for name,model in self.models.items():
            self.add_module(name, model)

        self.has_mean = False
        if 'mean_file' in self.net_info['props']:
            self.has_mean = True
            self.mean_file = self.net_info['props']['mean_file']

        self.blobs = None
        self.verbose = True
        self.train_outputs = []
        self.eval_outputs = []
        self.forward_data_only = False
        self.forward_net_only = False

    def set_forward_data_only(self, flag):
        self.forward_data_only = flag
        if self.forward_data_only:
            self.forward_net_only = False

    def set_forward_net_only(self, flag):
        self.forward_net_only = flag
        if self.forward_net_only:
            self.forward_data_only = False

    def set_forward_both(self):
        self.forward_data_only = False
        self.forward_net_only = False

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_phase(self, phase):
        self.phase = phase
        if phase == 'TRAIN':
            self.train()
        else:
            self.eval()

    def set_mean_file(self, mean_file):
        if mean_file != "":
            self.has_mean = True
            self.mean_file = mean_file
       
        else:
            self.has_mean = False
            self.mean_file = ""

    def set_train_outputs(self, *outputs):
        self.train_outputs = list(outputs)

    def set_eval_outputs(self, *outputs):
        self.eval_outputs = list(outputs)

    def get_outputs(self, outputs):
        blobs = []
        for name in outputs:
            blobs.append(self.blobs[name])
        return blobs

    def forward(self, *inputs): 
        if self.training:
            self.set_phase('TRAIN')
        else:
            self.set_phase('TEST')

        self.blobs = OrderedDict()
      
        if len(inputs) >= 2:
            data = inputs[0]
            label = inputs[1]
            self.blobs['data'] = data
            self.blobs['label'] = label
            if self.has_mean:
                nB = data.data.size(0)
                nC = data.data.size(1)
                nH = data.data.size(2)
                nW = data.data.size(3)
                data = data - Variable(self.mean_img.view(1, nC, nH, nW).expand(nB, nC, nH, nW))
        elif len(inputs) == 1:
            data = inputs[0]
            self.blobs['data'] = data
            if self.has_mean:
                nB = data.data.size(0)
                nC = data.data.size(1)
                nH = data.data.size(2)
                nW = data.data.size(3)
                data = data - Variable(self.mean_img.view(1, nC, nH, nW).expand(nB, nC, nH, nW))

        
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        self.output_loss = None
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if 'include' in layer and 'phase' in layer['include']:
                phase = layer['include']['phase']
                lname = lname + '.' + phase
                if phase != self.phase:
                    i = i + 1
                    continue
            ltype = layer['type']
            tname = layer['top']
            tnames = tname if type(tname) == list else [tname]
            if ltype in ['Data', 'AnnotatedData']:
                if (not self.omit_data_layer) and (not self.forward_net_only):
                    tdatas = self._modules[lname]()
                    if type(tdatas) != tuple:
                        tdatas = (tdatas,)
    
                    assert(len(tdatas) == len(tnames))
                    for index, tdata in enumerate(tdatas):
                        self.blobs[tnames[index]] = tdata
                    output_size = self.blobs[tnames[0]].size()
                    if self.verbose:
                        print('forward %-30s produce -> %s' % (lname, list(output_size)))
                i = i + 1
                continue

            if self.forward_data_only:
                break

            try:
                bname = layer['bottom']
            except:
                print(ltype)
                print(tname)
                print(lname)
                exit(0)
            bnames = bname if type(bname) == list else [bname]
            if True:
                bdatas = [self.blobs[name] for name in bnames]
                tdatas = self._modules[lname](*bdatas)
                if type(tdatas) != tuple:
                    tdatas = (tdatas,)

                assert(len(tdatas) == len(tnames))
                for index, tdata in enumerate(tdatas):
                    self.blobs[tnames[index]] = tdata
                if ltype in ['SoftmaxWithLoss', 'MultiBoxLoss']:
                    if self.output_loss != None:
                        self.output_loss += tdatas[0]
                    else:
                        self.output_loss = tdatas[0]
                i = i + 1
            input_size = self.blobs[bnames[0]].size()
            output_size = self.blobs[tnames[0]].size()
            if self.verbose:
                print('forward %-30s %s -> %s' % (lname, list(input_size), list(output_size)))

        #for name, blob in self.blobs.items():
        #    print('%s device = %d' % (name, blob.data.get_device()))
        
        if self.forward_data_only:
            odatas = [blob for blob in self.blobs.values()]
            return tuple(odatas)

        elif self.training:
            if len(self.train_outputs) > 1:
                odatas = [self.blobs[name] for name in self.train_outputs]
                return tuple(odatas)
            elif len(self.train_outputs) == 1:
                return self.blobs[self.train_outputs[0]]
            else:
                return self.blobs
        else:
            if len(self.eval_outputs) > 1:
                odatas = [self.blobs[name] for name in self.eval_outputs]
                return tuple(odatas)
            elif len(self.eval_outputs) == 1:
                return self.blobs[self.eval_outputs[0]]
            else:
                return self.blobs

    def get_loss(self):
        return self.output_loss

    def print_network(self):
        print(self)
        print_prototxt(self.net_info)

    def load_weights(self, caffemodel):
        if self.has_mean:
            print('mean_file', self.mean_file)
            mean_blob = caffe_pb2.BlobProto()
            mean_blob.ParseFromString(open(self.mean_file, 'rb').read())

            if 'input_shape' in self.net_info['props']:
                channels = int(self.net_info['props']['input_shape']['dim'][1])
                height = int(self.net_info['props']['input_shape']['dim'][2])
                width = int(self.net_info['props']['input_shape']['dim'][3])
            else:
                channels = int(self.net_info['props']['input_dim'][1])
                height = int(self.net_info['props']['input_dim'][2])
                width = int(self.net_info['props']['input_dim'][3])

            mu = np.array(mean_blob.data)
            mu.resize(channels, height, width)
            mu = mu.mean(1).mean(1)
            mean_img = torch.from_numpy(mu).view(channels, 1, 1).expand(channels, height, width).float()
            
            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)

        model = parse_caffemodel(caffemodel)
        layers = model.layer
        if len(layers) == 0:
            print('Using V1LayerParameter')
            layers = model.layers

        lmap = {}
        for l in layers:
            lmap[l.name] = l

        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if 'include' in layer and 'phase' in layer['include']:
                phase = layer['include']['phase']
                lname = lname + '.' + phase
            ltype = layer['type']
            if not lname in lmap:
                i = i + 1
                continue
            if ltype in ['Convolution', 'Deconvolution']:
                print('load weights %s' % lname)
                convolution_param = layer['convolution_param']
                bias = True
                if 'bias_term' in convolution_param and convolution_param['bias_term'] == 'false':
                    bias = False
                #weight_blob = lmap[lname].blobs[0]
                #print('caffe weight shape', weight_blob.num, weight_blob.channels, weight_blob.height, weight_blob.width)
                caffe_weight = np.array(lmap[lname].blobs[0].data)
                caffe_weight = torch.from_numpy(caffe_weight).view_as(self.models[lname].weight)
                self.models[lname].weight.data.copy_(caffe_weight)
                if bias and len(lmap[lname].blobs) > 1:
                    self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                    #print("convlution %s has bias" % lname)
                i = i + 1
            elif ltype == 'BatchNorm':
                print('load weights %s' % lname)
                self.models[lname].running_mean.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].running_var.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0]))

                i = i + 1
            elif ltype == 'PReLU':
                print('load weights %s' % lname)
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                i = i + 1
            elif ltype == 'Scale':
                print('load weights %s' % lname)
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'Normalize':
                print('load weights %s' % lname)
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                i = i + 1
            elif ltype == 'InnerProduct':
                print('load weights %s' % lname)
                if type(self.models[lname]) == nn.Sequential:
                    caffe_weight_size = self.models[lname][1].weight.data.size()
                    caffe_weight = np.array(lmap[lname].blobs[0].data).reshape(*caffe_weight_size)
                    caffe_weight = torch.from_numpy(caffe_weight)#.view_as(self.models[lname].weight)
                    self.models[lname][1].weight.data.copy_(caffe_weight)
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname][1].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                else:
                    self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            else:
                if not ltype in SUPPORTED_LAYERS:
                    print('load_weights: unknown type %s' % ltype)
                i = i + 1

    def create_network(self, net_info, input_width = None, input_height = None, input_channels = None):
        models = OrderedDict()
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()

        layers = net_info['layers']
        props = net_info['props']
        layer_num = len(layers)

        blob_channels['data'] = 3
        if input_channels != None:
            blob_channels['data'] = input_channels
        blob_height['data'] = 1
        if input_height != None:
            blob_height['data'] = input_height
        blob_width['data'] = 1
        if input_width != None:
            blob_width['data'] = input_width
        if 'input_shape' in props:
            blob_channels['data'] = int(props['input_shape']['dim'][1])
            blob_height['data'] = int(props['input_shape']['dim'][2])
            blob_width['data'] = int(props['input_shape']['dim'][3])
    
            self.width = int(props['input_shape']['dim'][3])
            self.height = int(props['input_shape']['dim'][2])
        elif 'input_dim' in props:
            blob_channels['data'] = int(props['input_dim'][1])
            blob_height['data'] = int(props['input_dim'][2])
            blob_width['data'] = int(props['input_dim'][3])
    
            self.width = int(props['input_dim'][3])
            self.height = int(props['input_dim'][2])

        if input_width != None and input_height != None:
            blob_width['data'] = input_width
            blob_height['data'] = input_height
            self.width = input_width
            self.height = input_height

        i = 0
        prelayer_type = ""
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if 'include' in layer and 'phase' in layer['include']:
                phase = layer['include']['phase']
                lname = lname + '.' + phase
            ltype = layer['type']
            tname = layer['top']
            if ltype in ['Data', 'AnnotatedData']:
                if not self.omit_data_layer:
                    models[lname] = CaffeData(layer.copy())
                    data, label = models[lname].forward()
                    data_name = tname[0] if type(tname) == list else tname
                    blob_channels[data_name] = data.size(1) # len(layer['transform_param']['mean_value'])
                    blob_height[data_name] = data.size(2) #int(layer['transform_param']['resize_param']['height'])
                    blob_width[data_name] = data.size(3) #int(layer['transform_param']['resize_param']['width'])
                    self.height = blob_height[data_name]
                    self.width = blob_width[data_name]
                i = i + 1
                continue
            try:
                bname = layer['bottom']
            except:
                print(ltype)
                print(lname)
                exit(0)
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_w'])
                stride = int(convolution_param['stride']) if 'stride' in convolution_param else 1
                pad = int(convolution_param['pad_w']) if 'pad_w' in convolution_param else 0
                group = int(convolution_param['group']) if 'group' in convolution_param else 1
                dilation = 1
                if 'dilation' in convolution_param:
                    dilation = int(convolution_param['dilation'])
                bias = True
                if 'bias_term' in convolution_param and convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, groups=group, bias=bias)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2*pad - kernel_size)//stride + 1
                blob_height[tname] = (blob_height[bname] + 2*pad - kernel_size)//stride + 1
                i = i + 1
            elif ltype == 'BatchNorm':
                momentum = 0.9
                if 'batch_norm_param' in layer and 'moving_average_fraction' in layer['batch_norm_param']:
                    momentum = float(layer['batch_norm_param']['moving_average_fraction'])
                if 'batch_norm_param' in layer and 'eps' in layer['batch_norm_param']:
                    eps = float(layer['batch_norm_param']['eps'])
                channels = blob_channels[bname]
                if prelayer_type == "InnerProduct":
                    models[lname] = nn.BatchNorm1d(channels, momentum=momentum, affine=False, eps=eps)
                else:
                    models[lname] = nn.BatchNorm2d(channels, momentum=momentum, affine=False, eps=eps)
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Scale':
                channels = blob_channels[bname]
                models[lname] = Scale(channels)
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'ReLU':
                inplace = (bname == tname)
                if 'relu_param' in layer and 'negative_slope' in layer['relu_param']:
                    negative_slope = float(layer['relu_param']['negative_slope'])
                    models[lname] = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
                else:
                    models[lname] = nn.ReLU(inplace=inplace)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'PReLU':
                models[lname] = nn.PReLU(blob_channels[bname])
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                padding = 0
                if 'pad' in layer['pooling_param']:
                    padding = int(layer['pooling_param']['pad'])
                pool_type = layer['pooling_param']['pool']
                if pool_type == 'MAX':
                    models[lname] = nn.MaxPool2d(kernel_size, stride, padding=padding, ceil_mode=True)
                elif pool_type == 'AVE':
                    models[lname] = nn.AvgPool2d(kernel_size, stride, padding=padding, ceil_mode=True)

                blob_width[tname] = int(math.ceil((blob_width[bname] + 2*padding - kernel_size)/float(stride))) + 1
                blob_height[tname] = int(math.ceil((blob_height[bname] + 2*padding - kernel_size)/float(stride))) + 1
                blob_channels[tname] = blob_channels[bname]
                i = i + 1
            elif ltype == 'Eltwise':
                operation = 'SUM'
                if 'eltwise_param' in layer and 'operation' in layer['eltwise_param']:
                    operation = layer['eltwise_param']['operation']
                bname0 = bname[0]
                bname1 = bname[1]
                models[lname] = Eltwise(operation)
                blob_channels[tname] = blob_channels[bname0]
                blob_width[tname] = blob_width[bname0]
                blob_height[tname] = blob_height[bname0]
                i = i + 1
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                if blob_width[bname] != -1 or blob_height[bname] != -1:
                    channels = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                    models[lname] = nn.Sequential(FCView(), nn.Linear(int(channels), int(filters)))
                else:
                    channels = blob_channels[bname]
                    models[lname] = nn.Linear(int(channels), int(filters))
                blob_channels[tname] = filters
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Dropout':
                channels = blob_channels[bname]
                dropout_ratio = float(layer['dropout_param']['dropout_ratio'])
                models[lname] = nn.Dropout(dropout_ratio, inplace=True)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Normalize':
                channels = blob_channels[bname]
                scale = float(layer['norm_param']['scale_filler']['value'])
                models[lname] = Normalize(channels, scale)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'LRN':
                local_size = int(layer['lrn_param']['local_size'])
                alpha = float(layer['lrn_param']['alpha'])
                beta = float(layer['lrn_param']['beta'])
                models[lname] = LRN(local_size, alpha, beta)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Permute':
                orders = layer['permute_param']['order']
                order0 = int(orders[0])
                order1 = int(orders[1])
                order2 = int(orders[2])
                order3 = int(orders[3])
                models[lname] = Permute(order0, order1, order2, order3)
                shape = [1, blob_channels[bname], blob_height[bname], blob_width[bname]]
                blob_channels[tname] = shape[order1]
                blob_height[tname] = shape[order2]
                blob_width[tname] = shape[order3]
                i = i + 1
            elif ltype == 'Flatten':
                axis = int(layer['flatten_param']['axis'])
                models[lname] = Flatten(axis)
                blob_channels[tname] = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Slice':
                axis = int(layer['slice_param']['axis'])
                assert(axis == 1)
                assert(type(tname) == list)
                slice_points = layer['slice_param']['slice_point']
                assert(type(slice_points) == list)
                assert(len(slice_points) == len(tname) - 1)
                slice_points = [int(s) for s in slice_points]
                shape = [1, blob_channels[bname], blob_height[bname], blob_width[bname]]
                slice_points.append(shape[axis])
                models[lname] = Slice(axis, slice_points)
                prev = 0
                for idx, tn in enumerate(tname):
                    blob_channels[tn] = slice_points[idx] - prev
                    blob_width[tn] = blob_width[bname]
                    blob_height[tn] = blob_height[bname]
                    prev = slice_points[idx]
                i = i + 1
            elif ltype == 'Concat':
                axis = 1
                if 'concat_param' in layer and 'axis' in layer['concat_param']:
                    axis = int(layer['concat_param']['axis'])
                models[lname] = Concat(axis)  
                if axis == 1:
                    blob_channels[tname] = 0
                    for bn in bname:
                        blob_channels[tname] += blob_channels[bn]
                        blob_width[tname] = blob_width[bn]
                        blob_height[tname] = blob_height[bn]
                elif axis == 2:
                    blob_channels[tname] = blob_channels[bname[0]]
                    blob_width[tname] = 1
                    blob_height[tname] = 0
                    for bn in bname:
                        blob_height[tname] += blob_height[bn]
                i = i + 1
            elif ltype == 'MultiBoxLoss':
                num_classes = int(layer['multibox_loss_param']['num_classes'])
                overlap_threshold = float(layer['multibox_loss_param']['overlap_threshold'])
                prior_for_matching = layer['multibox_loss_param']['use_prior_for_matching'] == 'true'
                bkg_label = int(layer['multibox_loss_param']['background_label_id'])
                neg_mining = True
                neg_pos = float(layer['multibox_loss_param']['neg_pos_ratio'])
                neg_overlap = float(layer['multibox_loss_param']['neg_overlap'])
                models[lname] = MultiBoxLoss(num_classes, overlap_threshold, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, use_gpu=True)
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Crop':
                axis = int(layer['crop_param']['axis'])
                offset = int(layer['crop_param']['offset'])
                models[lname] = Crop(axis, offset)
                blob_channels[tname] = blob_channels[bname[0]]
                blob_width[tname] = blob_width[bname[0]]
                blob_height[tname] = blob_height[bname[0]]
                i = i + 1
            elif ltype == 'Deconvolution':
                #models[lname] = nn.UpsamplingBilinear2d(scale_factor=2)
                #models[lname] = nn.Upsample(scale_factor=2, mode='bilinear')
                in_channels = blob_channels[bname]
                out_channels = int(layer['convolution_param']['num_output'])
                group = int(layer['convolution_param']['group'])
                kernel_w = int(layer['convolution_param']['kernel_w'])
                kernel_h = int(layer['convolution_param']['kernel_h'])
                stride_w = int(layer['convolution_param']['stride_w'])
                stride_h = int(layer['convolution_param']['stride_h'])
                pad_w = int(layer['convolution_param']['pad_w'])
                pad_h = int(layer['convolution_param']['pad_h'])
                kernel_size = (kernel_h, kernel_w)
                stride = (stride_h, stride_w)
                padding = (pad_h, pad_w)
                bias_term = layer['convolution_param']['bias_term'] != 'false'
                models[lname] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding=padding, groups = group, bias=bias_term)
                blob_channels[tname] = out_channels
                blob_width[tname] = 2 * blob_width[bname]
                blob_height[tname] = 2 * blob_height[bname]
                i = i + 1
            elif ltype == 'Reshape':
                reshape_dims = layer['reshape_param']['shape']['dim']
                reshape_dims = [int(item) for item in reshape_dims]
                models[lname] = Reshape(reshape_dims)
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Softmax':
                axis = 1
                if 'softmax_param' in layer and 'axis' in layer['softmax_param']:
                    axis = int(layer['softmax_param']['axis'])
                models[lname] = Softmax(axis)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            else:
                print('create_network: unknown type #%s#' % ltype)
                i = i + 1
            prelayer_type = ltype
            input_width = blob_width[bname] if type(bname) != list else blob_width[bname[0]]
            input_height = blob_height[bname] if type(bname) != list else blob_height[bname[0]]
            input_channels = blob_channels[bname] if type(bname) != list else blob_channels[bname[0]]
            output_width = blob_width[tname] if type(tname) != list else blob_width[tname[0]]
            output_height = blob_height[tname] if type(tname) != list else blob_height[tname[0]]
            output_channels = blob_channels[tname] if type(tname) != list else blob_channels[tname[0]]
            print('create %-20s (%4d x %4d x %4d) -> (%4d x %4d x %4d)' % (lname, input_channels, input_height, input_width, output_channels, output_height, output_width))

        return models

