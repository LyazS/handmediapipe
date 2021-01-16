old_parameter = {
    "num_layers": 5,
    "min_scale": 0.1171875,
    "max_scale": 0.75,
    "input_size_height": 256,
    "input_size_width": 256,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "fixed_anchor_size": True,
    "interpolated_scale_aspect_ratio": 1.0,
    "reduce_boxes_in_lowest_layer": False,
}
new_parameter = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "fixed_anchor_size": True,
    "interpolated_scale_aspect_ratio": 1.0,
    "reduce_boxes_in_lowest_layer": False,
}

import numpy as np


class Anchor():
    def __init__(self, cx=None, cy=None, wd=None, ht=None) -> None:
        super().__init__()
        self.cx = cx
        self.cy = cy
        self.wd = wd
        self.ht = ht

    def print(self):
        print(self.cx, self.cy, self.wd, self.ht)


def CalculateScale(min_scale, max_scale, stride_index, num_strides):
    return min_scale + (max_scale -
                        min_scale) * 1.0 * stride_index / (num_strides - 1)


def GenerateAnchor(option):
    anchors = []
    layer_id = 0
    while layer_id < len(option["strides"]):
        anchor_ht = []
        anchor_wd = []
        aspect_ratios = []
        scales = []

        last_same_stride_layer = layer_id
        while last_same_stride_layer < len(
                option["strides"]) and option["strides"][
                    last_same_stride_layer] == option["strides"][layer_id]:
            scale = CalculateScale(
                option["min_scale"],
                option["max_scale"],
                last_same_stride_layer,
                len(option["strides"]),
            )

            if last_same_stride_layer == 0 and option[
                    "reduce_boxes_in_lowest_layer"]:
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)

            else:
                for aspect_ratios_id in range(0, len(option["aspect_ratios"])):
                    aspect_ratios.append(
                        option["aspect_ratios"][aspect_ratios_id])
                    scales.append(scale)
                if option["interpolated_scale_aspect_ratio"] > 0.:
                    scale_next = CalculateScale(
                        option["min_scale"], option["max_scale"],
                        last_same_stride_layer + 1, len(option["strides"])
                    ) if last_same_stride_layer != len(
                        option["strides"]) - 1 else 1.0

                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(
                        option["interpolated_scale_aspect_ratio"])
            last_same_stride_layer += 1
        for i in range(0, len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_ht.append(scales[i] / ratio_sqrts)
            anchor_wd.append(scales[i] * ratio_sqrts)

        feature_map_ht = 0

        feature_map_wd = 0
        stride = option["strides"][layer_id]
        feature_map_ht = np.ceil(1. * option["input_size_height"] / stride)
        feature_map_wd = np.ceil(1. * option["input_size_width"] / stride)

        for y in range(0, int(feature_map_ht)):
            for x in range(0, int(feature_map_wd)):
                for anchor_id in range(0, len(anchor_ht)):
                    cx = (x + option["anchor_offset_x"] )* 1. / feature_map_wd
                    cy = (y + option["anchor_offset_y"] )* 1. / feature_map_ht

                    new_anchor = Anchor(cx, cy)
                    if option["fixed_anchor_size"]:
                        new_anchor.wd = 1.
                        new_anchor.ht = 1.
                    else:
                        new_anchor.wd = anchor_wd[anchor_id]
                        new_anchor.ht = anchor_ht[anchor_id]
                    anchors.append(new_anchor)
        layer_id = last_same_stride_layer
    return anchors


option = old_parameter
anchors = GenerateAnchor(option)
print(len(anchors))
with open("model/anchors_old.csv","w") as f:
    
    for a in anchors:
        f.write("{0},{1},{2},{3}\n".format(a.cx,a.cy,a.wd,a.ht))