## Dataset
Files downloaded from https://vision.soic.indiana.edu/projects/egohands/ under Caffe Models. Choose
from "Hand classification/detection network" and download inside of ```utils/caffe_model``` folder.

Rename ``file.prototxt`` file to ``hands_detector.prototxt`` and ``file.caffemodel`` to ``hand_detector.prototxt``

## Altering .prototxt

Remove the two following layers as they don't exist in pytorch:


```
#WINDOW_DATA layer contains preprocessing information needed for torch transformations

layers {
  name: "data"
  type: WINDOW_DATA
  top: "data"
  top: "label"
  window_data_param {
    source: "text_file_accroding_to_window_data_layer_formatting.txt"
    batch_size: 256
    fg_threshold: 0.5
    bg_threshold: 0.5
    context_pad: 16
    crop_mode: "warp"
  }
  transform_param {
    crop_size: 227
    mean_file: "/path_to_caffe/data/ilsvrc12/imagenet_mean.binaryproto"
    mirror: false
  }
  include { 
    phase: TEST 
  }
}
```


```
#ACCURACY layer is equivalent of doing model.eval() and using F.accuracy()


layers {
  name: "accuracy"
  type: ACCURACY
  bottom: "prob"
  bottom: "label"
  top: "accuracy"
  include { 
    phase: TEST 
  }
}
```

## Minimal Working example
Available in ```caffe2pytorch.ipynb```