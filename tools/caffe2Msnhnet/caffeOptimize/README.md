# caffeOptimize

caffe的图优化工具，已支持操作

- [x] Conv+BN+Scale 融合到 Conv
- [x] Deconv+BN+Scale 融合到Deconv
- [x] InnerProduct+BN+Scale 融合到InnerProduct

# 使用方法

python caffeOptimize.py --model landmark106.prototxt --weights landmark106.caffemodel
