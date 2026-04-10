# `simaaiboxdecodebf16` -- per-model JSON guide

Pick the row matching your model in the matrix below, copy the named
template from `cfg/templates/`, and edit the four "Customer changes here"
fields. Everything else has working defaults.

## Step 1 -- pick the right template

| Your model                                           | Template file                                    | `decode_type`                          | num tensors |
|------------------------------------------------------|--------------------------------------------------|----------------------------------------|------|
| **YOLOv8 / v9 / v10 / v11 / v12 / yolo26**, post graph-surgery (cxcywh) | `templates/anchorfree_surgeon_depth4.json`       | `yolov8` *(or your family name)*       | 6    |
| Same families, **raw DFL bins** (no graph surgery)   | `templates/anchorfree_dfl_depth64.json`          | `yolov8` *(or your family name)*       | 6    |
| **YOLOv5 / v7** anchor-based                         | `templates/anchorbased_v5_v7.json`               | `yolov5` or `yolov7`                   | 3    |
| **YOLOX**                                            | `templates/yolox.json`                           | `yolox`                                | 3    |
| **YOLOv8-pose** / **v11-pose**                       | (use anchor-free template + pose extras -- see below) | `yolov8-pose` / `yolov11-pose`     | 9    |
| **YOLOv8/9/10/11-seg**                               | (anchor-free + seg extras)                       | `yolov8-seg` *(etc.)*                  | 10   |
| **YOLOv5-seg / v7-seg**                              | (anchor-based + seg extras)                      | `yolov5-seg` / `yolov7-seg`            | 4    |

## Step 2 -- customer changes per template

These are the only fields a customer normally edits. Everything else in
the templates is already correct.

| Field                  | What to set it to                                                           |
|------------------------|-----------------------------------------------------------------------------|
| `decode_type`          | Exact family name from the table above (no aliases).                        |
| `num_classes`          | Your model's class count (COCO=80, custom datasets vary).                   |
| `model_width` / `model_height` | The model's input resolution (e.g. 640x640).                        |
| `original_width` / `original_height` | The video frame resolution feeding preproc (e.g. 1280x720).   |
| `class_is_prob`        | `true` if the model graph applies sigmoid to class outputs (typical SiMa surgeon export); `false` if the model emits raw logits. |
| `input_depth[]`        | See "Input-depth recipes" below -- depends on family + DFL/no-DFL.          |
| `input_height[] / input_width[]` | The H,W of each head. For 640x640 input these are `[80,40,20]` (strides 8/16/32). |
| `topk`                 | Max kept detections per frame (default 24).                                 |
| `detection_threshold`  | Confidence cutoff (default 0.5).                                            |
| `nms_iou_threshold`    | NMS IoU cutoff (default 0.3).                                               |
| `buffers.input[0].size` | Bytes the upstream `simaaiprocessdetessellate_1` produces -- set per family.   |
| `buffers.output.size`  | `4 + topK * 24` for bbox (= 580 at topK=24).                                |

## Step 3 -- Input-depth recipes

Notation: `nc` = `num_classes`, `na` = `num_anchors` (default 3 for v5/v7).

| Family            | `num_in_tensor` | `input_depth[]`                                               |
|-------------------|-----------------|---------------------------------------------------------------|
| Anchor-free, surgeoned (depth=4) | 6 | `[4,  4,  4,  nc, nc, nc]`                            |
| Anchor-free, DFL  (depth=64) | 6 | `[64, 64, 64, nc, nc, nc]`                                |
| YOLOv5 / v7       | 3               | `[na*(5+nc), na*(5+nc), na*(5+nc)]` (= 255 for na=3, nc=80)   |
| YOLOX             | 3               | `[4+1+nc, 4+1+nc, 4+1+nc]`        (= 85  for nc=80)            |
| YOLOv8-pose / v11-pose | 9          | `[4or64, 4or64, 4or64, nc, nc, nc, 3*kp, 3*kp, 3*kp]`         |
| YOLOv8/9/10/11-seg | 10             | `[4or64, ..., nc, nc, nc, 32, 32, 32, proto_size]`            |
| YOLOv5/v7-seg     | 4               | `[na*(5+nc+32), x3,  proto_size]`                             |

The plugin auto-handles the `4 vs 64` choice for the bbox path -- if any
bbox tensor depth is `>= 64` it switches to softmax-EV / DFL decoding.

## Step 4 -- letterbox / padding (almost always default)

The SiMa preprocessor's "CENTER" padding actually places only **1/4** of the
vertical gap on top (not 1/2). For 1280x720 -> 640x640 the defaults the
plugin computes are:

```
gain       = 0.5
pad_left   = 0,   pad_right = 0
pad_top    = (640 - 720*0.5) / 4 = 70
pad_bottom = 280 - 70           = 210
```

These are correct for the standard SiMa preproc; **don't override unless
you calibrated otherwise**. JSON keys to override:

```json
"pad_top": 70, "pad_bottom": 210, "pad_left": 0, "pad_right": 0,
"letterbox_gain": 0.5
```

(`-1` or omitted = "use computed default".)

## Step 5 -- bbox format flag (depth=4 only)

When `input_depth[0] == 4` the bbox tensor's 4 channels can be one of two
things, depending on how the model was surgeoned:

| `bbox_format`    | meaning                                                            | when to use |
|------------------|--------------------------------------------------------------------|-------------|
| `cxcywh_pixel` (default) | already-decoded `[cx, cy, w, h]` in model-space pixels -- anchor + stride baked in | SiMa `surgeon_yolov8` and most graph-surgery'd v8/v9/v11/v12/yolo26 exports |
| `ltrb_grid`      | raw `[l, t, r, b]` distances in grid units -- apply anchor + stride in C++ | Exotic surgeons that fold DFL but skip the anchor add |

For `input_depth[0] == 64` (DFL bins) this flag is ignored.

## Step 6 -- verify after running

The plugin prints exactly which path it took. Look for this line in stdout:

```
[yolodecode] decode=yolov8 render=bbox classes=80 ... bbox_dfl=0
   bbox_format=cxcywh_pixel class_is_prob=1 strides=[8,16,32]
   letterbox=[gain=0.5 pad_top=70 pad_bottom=210 pad_left=0 pad_right=0]
   out_buf=580
```

If `decode_type` is wrong you'll see an error listing every accepted name.
If tensor counts mismatch you'll see a `WARNING: ... expects N tensors,
JSON has M`.

## Step 7 -- output buffer size formulas

| `render_type` | output size formula (bytes)                  | example (topK=24, nc=80, kp=17) |
|---------------|----------------------------------------------|----------------------------------|
| `bbox`        | `4 + topK * 24`                              | `580`                            |
| `bboxs`       | same as `bbox` (uses score slot for label)   | `580`                            |
| pose          | `4 + topK * 24 + topK * num_keypoints * 12`  | `5476`                           |
| seg           | `4 + topK * (24 + 160 * 160)`                | `615964`                         |

Set `buffers.output.size` accordingly. The render plugin reads exactly that
many bytes.
