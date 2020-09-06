import numpy as np 
from MsnhBuilder import Msnhnet
import math
from struct import pack

def getPadding(config_keras, input_shape):
    if config_keras['padding']=='valid':
        return [0,0]
    
    if config_keras['padding']=='same':
        if 'kernel_size' in config_keras:
            kernel_size = config_keras['kernel_size']
        elif 'pool_size' in config_keras:
            kernel_size = config_keras['pool_size']
        else:
            raise Exception('Undefined kernel size')
        
        strides = config_keras['strides']
        w = input_shape[1]
        h = input_shape[2]
        
        out_w = math.floor(w / float(strides[1]))
        pad_w = int((kernel_size[1]*out_w - (kernel_size[1]-strides[1])*(out_w - 1) - w)/2)
        
        out_h = math.floor(h / float(strides[0]))
        pad_h = int((kernel_size[0]*out_h - (kernel_size[0]-strides[0])*(out_h - 1) - h)/2)

        if pad_w==0 and pad_h==0:
            return [0,0]
        
        return [pad_w, pad_h]
    else:
        raise Exception(config_keras['padding']+' padding is not supported')

def keras2Msnh(keras_model, msnhnet_file, msnhbin_file):

    net = Msnhnet()
    params = [] 

    i = 0
    lastname = ''   # last layer output name(name)

    for layer in keras_model.layers:

        if type(layer.output)==list: # multi out is not support
            raise Exception(' mult out is not support') 

        name = layer.output.name    # output name

        if type(layer.input)!=list:  # current layer's prev layer

            input_ = layer.input.name   # name of prev layer

            if i != 0: # first layer is config
                if input_ != lastname : # if prev layer is not last layer, a route layer is needed 
                    net.buildRoute(name, str(net.name_index_dict[input_]), 0) # add a route layer
        
        i = i + 1

        lastname = name # last layer name 

        layer_type = type(layer).__name__ # layer type

        config = layer.get_config() # config

        print(config)

        blobs = layer.get_weights() # weights

        if layer_type=='InputLayer' :  # Input layer -> Config layer
            input_shape = config['batch_input_shape']
            shape = [1, input_shape[3],input_shape[1],input_shape[2]]
            net.buildConfig(shape)

        elif layer_type=='Conv2D' or layer_type=='Convolution2D' or layer_type=='DepthwiseConv2D': # Conv2d DepthWiseConv2D
            strides = config['strides']
            kernel_size = config['kernel_size']
            filters = config['filters']
            dilation = config['dilation_rate']

            if not config['use_bias']:
                useBias = 0
            else:
                useBias = 1

            if layer_type == 'DepthwiseConv2D':
                groups = layer.input_shape[3]
            else:
                groups = 1

            padding = getPadding(config, layer.input_shape)

            net.buildConv2d(name, filters, kernel_size[0], kernel_size[1], padding[0], padding[1], \
                            strides[0], strides[1], dilation[0], dilation[1], groups, useBias)
            
            if config['activation'] == "softmax":
                net.buildSoftmax(name)
            else:
                if config['activation'] != "linear" :
                    net.buildActivation(name,config['activation'])

            if layer_type == 'DepthwiseConv2D':
                blobs[0] = np.array(blobs[0]).transpose(2,3,0,1)
            else:
                blobs[0] = np.array(blobs[0]).transpose(3,2,0,1)

            blobs[1] = np.array(blobs[1])
            params.extend(blobs[0].flatten().tolist())
            params.extend(blobs[1].flatten().tolist())

        elif layer_type=='MaxPooling2D' or layer_type=='AveragePooling2D': # Pool layer
            pool_size = config['pool_size']
            
            strides  = config['strides']    

            padding = getPadding(config, layer.input_shape)

            if layer_type=='MaxPooling2D':
                type_ = "MAX"
            else:
                type_ = "AVE"
            
            net.buildPooling(name, type_, pool_size[0], pool_size[1], strides[0], strides[1], padding[0], padding[1])
                
        elif layer_type=='BatchNormalization': # Bn layer
            if not config['scale'] :
                raise Exception('Bn not support ')

            if not config['center'] :
                raise Exception('Bn not support ')

            blobs[0] = np.array(blobs[0]) # scale
            blobs[1] = np.array(blobs[1]) # beta / biase
            blobs[2] = np.array(blobs[2]) # mean
            blobs[3] = np.array(blobs[3]) # variance
            params.extend(blobs[0].flatten().tolist())
            params.extend(blobs[1].flatten().tolist())
            params.extend(blobs[2].flatten().tolist())
            params.extend(blobs[3].flatten().tolist())
            net.buildBatchNorm(name)

        elif layer_type=='LeakyReLU':
            net.buildActivation(name,'leakylu')

        elif layer_type=='Activation':
            if config['activation'] == 'softmax':
                net.buildSoftmax(name)
            else:
                net.buildActivation(name,config['activation'])

        elif layer_type=='UpSampling2D':
            if config['size'][0]!=config['size'][1]:
                raise Exception('Unsupported upsampling factor')

            if config['interpolation']!='nearest':
                raise Exception('only nearest is supported by Upsample')
            stride = config['size'][0]
            net.buildUpsample2D(name,stride)

        elif layer_type=='Concatenate' or layer_type=='Merge' or layer_type=='Add':
            layers = ''
            for l in layer.input:
                layers = layers + str(int(net.name_index_dict[l.name])) + ','
            layers = layers[0:-1]
            if layer_type == 'Add' :
                addModel = 1
            else:
                addModel = 0
            net.buildRoute(name, layers, addModel)

        elif layer_type=='ZeroPadding2D':
            top = config['padding'][0][0]
            down = config['padding'][0][1]
            left = config['padding'][1][0]
            right = config['padding'][1][1]
            net.buildPadding(name, top, down, left, right)

        elif layer_type=='GlobalAveragePooling2D':
            net.buildGlobalAvgPooling(name)

        elif layer_type=='Dense':
            output = config['units']
            net.buildConnect(name,output)

            if config['activation'] == "softmax":
                net.buildSoftmax(name)
            else:
                net.buildActivation(name, config['activation'])

            if config['use_bias']:
                weight=np.array(blobs[0]).transpose(1, 0)
                if type(layer._inbound_nodes[0].inbound_layers[0]).__name__=='Flatten':
                    flatten_shape=layer._inbound_nodes[0].inbound_layers[0].input_shape
                    for i in range(weight.shape[0]):
                        weight[i]=np.array(weight[i].reshape(flatten_shape[1],flatten_shape[2],flatten_shape[3]).transpose(2,0,1).reshape(weight.shape[1]))
                
                blobs[1] = np.array(blobs[1])
                params.extend(weight.flatten().tolist())
                params.extend(blobs[1].flatten().tolist())
            else:
                blobs[0] = np.array(blobs[0])
                params.extend(blobs[0].flatten().tolist())
        elif layer_type=='Flatten' or layer_type=='Flatten' or layer_type=='Reshape':
            pass
        else:
            raise Exception(layer_type + ' not support ')
        
    with open(msnhnet_file,"w") as f1:
        f1.write(net.net)

    with open(msnhbin_file,"wb") as f:
        for i in params :
            f.write(pack('f',i))