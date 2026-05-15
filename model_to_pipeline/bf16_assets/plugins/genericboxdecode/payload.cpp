//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
//**************************************************************************
//
// Generic BFLOAT16 YOLO decoder (single-file plugin, no external `run`).
//
//  Pipeline matched to this folder's preprocessing config:
//      preproc: 1280x720 NV12 -> 640x640 RGB BF16, CENTER pad, BILINEAR resize
//      mla    : YOLOv8m / v9 / v10 / v11 (anchor-free), 6 output tensors
//      detess : tessellated MLA output -> NHWC contiguous BF16
//      THIS   : BF16 -> FP32 -> (NHWC -> NCHW) -> decode -> [N][topK boxes]
//      render : reads buffer as render's `Points` (24B / box, classId only)
//
//  Supported decode_type values (read from JSON; customer must set explicitly):
//      "yolov8" / "yolov9" / "yolov10" / "yolov11" /
//      "yolov12" / "yolo26"                                  -> bbox detection (anchor-free, 6 tensors)
//      "yolov5" / "yolov7"                                   -> bbox (anchor-based, 3 tensors)
//      "yolox"                                               -> bbox detection (3 tensors)
//      "yolov8-pose"  / "yolov11-pose"                       -> pose
//      "yolov8-seg"   / "yolov9-seg"  /
//      "yolov10-seg"  / "yolov11-seg"                        -> segmentation
//      "yolov5-seg"   / "yolov7-seg"                         -> segmentation (anchor-based)
//
//  Anchor-free family (v8 / v9 / v10 / v11 / v12 / v26) all share the same
//  decode path. Architectural differences (CSP / ELAN / attention / NMS-free
//  dual heads) are confined to the model graph; the post-processing format
//  (3 bbox heads + 3 cls heads, NHWC) is identical. depth=4 (post graph-
//  surgery cxcywh) and depth=64 (raw DFL bins) are both handled via the
//  bbox_is_dfl heuristic -- the customer only changes input_depth in JSON.
//
//  See cfg/MODELS.md and the cfg/templates/ directory for per-model JSON
//  templates a customer can copy and adapt.
//
//  Input buffer layout
//  -------------------
//  Single contiguous BF16 (uint16) NHWC buffer; tensors concatenated in
//  the order described by JSON arrays input_depth/input_height/input_width.
//  For tensor t with shape [H, W, C], element (y, x, c) sits at offset
//  y*W*C + x*C + c (channels-last). The detessellate plugin emits this.
//
//  Tensor ordering convention
//  --------------------------
//      Anchor-free detection (6 tensors):  bbox_0..2 | cls_0..2
//      Anchor-free pose      (9 tensors):  bbox_0..2 | cls_0..2 | kp_0..2
//      Anchor-free seg      (10 tensors):  bbox_0..2 | cls_0..2 | mc_0..2 | proto
//      Anchor-based det      (3 tensors):  per-scale [H, W, na*(5+num_classes)]
//      Anchor-based seg      (4 tensors):  per-scale [H, W, na*(5+num_classes+32)] | proto
//      YOLOX                 (3 tensors):  per-scale [H, W, 4 + 1 + num_classes]
//
//  Stage 1: BF16 -> FP32 (NEON SIMD)
//  Stage 2: NHWC -> NCHW transpose (per-tensor) into a contiguous scratch.
//           Post-processing references each tensor's NCHW base directly.
//           Indexing: element (c, y, x) of tensor t -> nchw[c*H*W + y*W + x].
//  Stage 3: Anchor-free / anchor-based / yolox / pose / seg decode,
//           per-class greedy NMS, top-K keep.
//  Stage 4: Inverse symmetric-CENTER letterbox to original frame coords;
//           pack into the render plugin's expected layout.
//
//  Output buffer layouts (consumed by `simaairender` from boxrender.h)
//  -------------------------------------------------------------------
//    BBox          : [int num_det][topK x BoundingBoxOut(24B)]
//                      Total = 4 + topK * 24      (e.g. topK=24 -> 580 B)
//    Pose          : [int num_det][topK x BoundingBoxOut(24B)]
//                                 [topK x (num_keypoints x PosePoint(12B))]
//                      Total = 4 + topK*24 + topK*num_keypoints*12
//    Segmentation  : [int num_det][topK x (BoundingBoxOut(24B) + uint8 mask[160*160])]
//                      Total = 4 + topK * (24 + 160*160)
//
//  BoundingBoxOut is byte-compatible with render's `Points` struct:
//      x1, y1, w, h, _score (=trackId-slot, used by "bboxs"), class_id
//
//  bbox tensor depth heuristic (anchor-free family):
//      input_depth[0] == 4   -> already-decoded distances [l, t, r, b] (grid units)
//      input_depth[0] == 64  -> raw DFL bins (4 sides x 16, softmax-EV applied)
//
//  Class output heuristic
//  ----------------------
//  Many SiMa-converted YOLO models name the class head `class_prob_*` and
//  bake the sigmoid into the model. In that case the values are already
//  probabilities in [0, 1]; applying sigmoid again would crush everything
//  toward 0.5..0.73 and produce wrong detections.
//  JSON flag `class_is_prob` (default false) controls this:
//      class_is_prob = false (default): model emits raw logits  -> sigmoid in C++
//      class_is_prob = true           : model emits probabilities -> no sigmoid
//
//  INT8 quantized models are NOT supported here. Use the original
//  `simaai_boxdecode` for those (it has the proven TesselatedTensor reader).
//
//**************************************************************************

#include <aggregator/agg_template.h>
#include <arm_neon.h>
#include <vector>
#include <span>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

//==============================================================================
//  Output structs (byte-exact contract with the render plugin)
//==============================================================================
//  Layout (24 bytes) is a 1:1 match with `Points` in boxrender's cv_utils.h:
//      uint32 x1; uint32 y1; uint32 w; uint32 h; uint32 trackId; uint32 classId
//  We store the score in the trackId slot. RENDER_TYPE::BOX ignores it;
//  RENDER_TYPE::BOXS reinterpret_casts it to float for label-with-score.
struct BoundingBoxOut {              // 24 bytes
    uint32_t _x;
    uint32_t _y;
    uint32_t _w;
    uint32_t _h;
    float    _score;
    uint32_t _class_id;
};

struct PosePoint {                   // 12 bytes
    uint32_t _x;
    uint32_t _y;
    float    _visible;
};

constexpr int DETECTION_HEAD_COUNT = 3;       // 80x80, 40x40, 20x20
constexpr int DEFAULT_STRIDES[3]   = {8, 16, 32};
constexpr int DFL_REG_MAX          = 16;      // bins per side for DFL
constexpr int DFL_BBOX_CH          = 64;      // 4 * 16
constexpr int MASK_PROTO_CH        = 32;      // mask coefficient length
constexpr int MASK_W               = 160;
constexpr int MASK_H               = 160;
constexpr int MASK_PIXELS          = MASK_W * MASK_H;

//==============================================================================
//  Decoder dispatch
//==============================================================================
enum class DecodeKind {
    YOLO_DET,         // yolov8 / v9 / v10 / v11 detection (anchor-free)
    YOLO_DET_ANCHOR,  // yolov5 / v7 detection (anchor-based)
    YOLOX_DET,        // yolox detection
    YOLO_POSE,        // yolov8-pose / yolov11-pose
    YOLO_SEG,         // yolov8/9/10/11 - seg
    YOLO_SEG_ANCHOR,  // yolov5-seg / yolov7-seg
    UNKNOWN
};

// Customers must set `decode_type` explicitly. The mapping below is the only
// supported list -- any other value is rejected with a clear error message
// that names every accepted choice.
//
// Detection family routing (anchor-free, 6 tensors: 3 bbox + 3 cls):
//   yolov8 / yolov9 / yolov10 / yolov11 / yolov12 / yolo26
// Anchor-based (3 tensors, packed na*(5+nc) per cell):
//   yolov5 / yolov7
// YOLOX (3 tensors, packed 4+1+nc per cell, no anchors):
//   yolox
// All anchor-free variants share the same code path; the differences between
// v8 / v9 / ... / v26 (CSP vs ELAN vs attention etc.) are confined to the
// model graph itself, not the post-processing format. Both depth=4 (post
// graph-surgery cxcywh_pixel) and depth=64 (raw DFL bins) are handled via
// the bbox_is_dfl heuristic in the constructor; no JSON change needed.
static DecodeKind kind_from_string(const string& s) {
    if (s == "yolov8"  || s == "yolov9"  || s == "yolov10" ||
        s == "yolov11" || s == "yolov12" || s == "yolo26")                    return DecodeKind::YOLO_DET;
    if (s == "yolov5"  || s == "yolov7")                                      return DecodeKind::YOLO_DET_ANCHOR;
    if (s == "yolox")                                                         return DecodeKind::YOLOX_DET;
    if (s == "yolov8-pose"  || s == "yolov11-pose")                           return DecodeKind::YOLO_POSE;
    if (s == "yolov8-seg"   || s == "yolov9-seg" ||
        s == "yolov10-seg"  || s == "yolov11-seg")                            return DecodeKind::YOLO_SEG;
    if (s == "yolov5-seg"   || s == "yolov7-seg")                             return DecodeKind::YOLO_SEG_ANCHOR;
    return DecodeKind::UNKNOWN;
}

//==============================================================================
//  Plugin configuration loaded from JSON
//==============================================================================
struct PluginCfg {
    DecodeKind  kind            = DecodeKind::UNKNOWN;
    string      decode_type_str;
    string      render_type_str;

    int         num_classes     = 80;
    int         num_keypoints   = 17;
    int         model_w         = 640;
    int         model_h         = 640;
    int         orig_w          = 1280;
    int         orig_h          = 720;

    int         topk            = 24;
    float       conf_thresh     = 0.5f;
    float       nms_thresh      = 0.5f;

    int         num_in_tensor   = 0;
    vector<int> tensor_d, tensor_h, tensor_w;
    size_t      output_size     = 0;

    bool        bbox_is_dfl     = false;
    bool        class_is_prob   = false;        // skip sigmoid on class head
    // BBox 4-channel format (when input_depth[0] == 4):
    //   "cxcywh_pixel" (default for depth=4): SiMa surgeon_yolov8 bakes anchor +
    //                  stride into the graph -> 4 channels are [cx, cy, w, h]
    //                  in model-space pixels, ready to use as-is.
    //   "ltrb_grid"  : raw [l, t, r, b] distances in grid units (apply
    //                  anchor + stride multiplication in C++).
    // For input_depth[0] == 64 the DFL path runs and this is ignored.
    string      bbox_format     = "cxcywh_pixel";

    // Inverse letterbox overrides.
    //
    // The SiMa preprocessor labels its padding "CENTER", but it does NOT split
    // the vertical gap evenly: it puts only one quarter of the total vertical
    // gap on top (and three quarters on the bottom). Reference impl in
    // simaaiyolodecode/payload.cpp documents this and uses
    //
    //     pad_x_default = (model_w - frame_w * gain) / 2     (true center)
    //     pad_y_default = (model_h - frame_h * gain) / 4     (SiMa quirk)
    //
    // For 1280x720 -> 640x640 (gain = 0.5):
    //     total_v_gap = 640 - 720*0.5 = 280
    //     pad_top default = 280 / 4 = 70
    //
    // Subtracting the textbook 140 over-corrects by 70 model-px (= 140 frame-
    // px), which makes every box land 140px too high on the rendered frame --
    // the exact "boxes above expected region" symptom. The /4 factor is the
    // empirical fix matched against the SiMa preproc output.
    //
    // If the JSON sets any of `pad_top`, `pad_bottom`, `pad_left`, `pad_right`,
    // those override the above. -1 = "use computed default". The y inverse
    // uses pad_top, the x inverse uses pad_left.
    float       pad_top         = -1.f;
    float       pad_bottom      = -1.f;
    float       pad_left        = -1.f;
    float       pad_right       = -1.f;
    // Explicit gain override (rare; e.g. when the model was trained on a
    // direct-stretch resize rather than aspect-preserving letterbox).
    float       gain_override   = -1.f;

    int         strides[DETECTION_HEAD_COUNT] = {8, 16, 32};
    int         heads_h[DETECTION_HEAD_COUNT] = {0, 0, 0};
    int         heads_w[DETECTION_HEAD_COUNT] = {0, 0, 0};

    // Anchor-based heads (yolov5 / v7). 3 scales x 3 anchors x (w, h) px.
    int         num_anchors     = 3;
    float       anchors_wh[DETECTION_HEAD_COUNT][3][2] = {
        // COCO defaults: P3/8, P4/16, P5/32  - override via JSON "anchors".
        { {10.f,13.f},  {16.f, 30.f},  {33.f, 23.f} },
        { {30.f,61.f},  {62.f, 45.f},  {59.f,119.f} },
        { {116.f,90.f}, {156.f,198.f}, {373.f,326.f} },
    };

    bool        debug_log       = false;
    bool        initialized     = false;
};

static PluginCfg     g_cfg;
static vector<float> g_fp32_nhwc;   // BF16 -> FP32 scratch (still NHWC, all tensors)
static vector<float> g_fp32_nchw;   // NHWC -> NCHW scratch (all tensors back-to-back)
static vector<int>   g_tensor_off;  // per-tensor offsets into g_fp32_nchw

//==============================================================================
//  Helpers
//==============================================================================
static inline int tensor_size(const PluginCfg& c, int t) {
    return c.tensor_d[t] * c.tensor_h[t] * c.tensor_w[t];
}

// BF16 -> FP32 (NEON, 8 elements/cycle). Drop-in for the bf16-to-fp32 cast.
static inline void bf16_to_fp32(const uint16_t* src, float* dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v  = vld1q_u16(src + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(v),  16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
    for (; i < n; ++i) {
        uint32_t b = static_cast<uint32_t>(src[i]) << 16;
        memcpy(&dst[i], &b, 4);
    }
}

// Per-tensor NHWC -> NCHW transpose.
//   src[y*W*C + x*C + c]  ->  dst[c*H*W + y*W + x]
// Single buffer per call; H, W, C come from the tensor shape.
// We keep the inner loop on `c` (channel) read-contiguous, then scatter to
// `dst` strided by H*W -- this is the cheaper of the two access patterns
// because the strided writes remain cache-line aligned.
static void nhwc_to_nchw_one(const float* src, float* dst, int H, int W, int C) {
    const int HW = H * W;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float* in  = src + (y * W + x) * C;
            float*       out = dst + (y * W + x);
            for (int c = 0; c < C; ++c) {
                out[c * HW] = in[c];
            }
        }
    }
}

// Transpose every model tensor from NHWC into a single NCHW scratch buffer.
// Tensors are stored back-to-back in NCHW order; offsets cached in g_tensor_off.
static void transpose_all_to_nchw(const PluginCfg& c) {
    int off_in  = 0;
    int off_out = 0;
    for (int t = 0; t < c.num_in_tensor; ++t) {
        const int H = c.tensor_h[t];
        const int W = c.tensor_w[t];
        const int C = c.tensor_d[t];
        const int N = H * W * C;
        g_tensor_off[t] = off_out;
        nhwc_to_nchw_one(g_fp32_nhwc.data() + off_in,
                         g_fp32_nchw.data() + off_out,
                         H, W, C);
        off_in  += N;
        off_out += N;
    }
}

static inline float sigmoid_f(float x) {
    if (x >  50.f) return 1.f;
    if (x < -50.f) return 0.f;
    return 1.f / (1.f + expf(-x));
}

// Numerically-stable softmax over REG_MAX bins -> expected value.
static inline float dfl_decode(const float* bins, int reg_max = DFL_REG_MAX) {
    float mx = bins[0];
    for (int k = 1; k < reg_max; ++k) if (bins[k] > mx) mx = bins[k];
    float sum = 0.f, val = 0.f;
    for (int k = 0; k < reg_max; ++k) {
        float e = expf(bins[k] - mx);
        sum += e;
        val += static_cast<float>(k) * e;
    }
    return (sum > 0.f) ? val / sum : 0.f;
}

// Internal candidate (model-space cx, cy, w, h).
struct RawDet {
    float cx, cy, w, h;
    float score;
    int   label;
    int   head, y_cell, x_cell;                 // for pose/seg lookups
    array<float, MASK_PROTO_CH> mcoef{};        // mask coeffs (seg only)
};

static float iou_cxcywh(const RawDet& a, const RawDet& b) {
    float ax1 = a.cx - a.w * 0.5f, ay1 = a.cy - a.h * 0.5f;
    float ax2 = a.cx + a.w * 0.5f, ay2 = a.cy + a.h * 0.5f;
    float bx1 = b.cx - b.w * 0.5f, by1 = b.cy - b.h * 0.5f;
    float bx2 = b.cx + b.w * 0.5f, by2 = b.cy + b.h * 0.5f;
    float iw = max(0.f, min(ax2, bx2) - max(ax1, bx1));
    float ih = max(0.f, min(ay2, by2) - max(ay1, by1));
    float inter = iw * ih;
    float uni   = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return (uni > 0.f) ? inter / uni : 0.f;
}

static void per_class_nms(vector<RawDet>& dets, float iou_thr) {
    sort(dets.begin(), dets.end(),
         [](const RawDet& a, const RawDet& b){ return a.score > b.score; });
    vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].label != dets[j].label) continue;
            if (iou_cxcywh(dets[i], dets[j]) > iou_thr) suppressed[j] = true;
        }
    }
    vector<RawDet> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i)
        if (!suppressed[i]) kept.push_back(dets[i]);
    dets.swap(kept);
}

// Inverse letterbox parameters.
//   gain      : model_pixel = orig_pixel * gain  (for the resized image area)
//   pad_top   : model rows above the resized image
//   pad_bottom: model rows below the resized image
//   pad_left  : model columns left  of the resized image
//   pad_right : model columns right of the resized image
//
//   Inverse: orig_y = (model_y - pad_top)  / gain
//            orig_x = (model_x - pad_left) / gain
//
// SiMa preproc quirk (matches reference impl simaaiyolodecode/payload.cpp):
// the "CENTER" padding is centered on x but NOT on y. The vertical gap is
// split 1/4 on top, 3/4 on bottom. So for 1280x720 -> 640x640 the defaults are
//     gain        = 0.5
//     pad_left    = 0
//     pad_right   = 0
//     pad_top     = (640 - 720*0.5) / 4 = 70    <-- /4, not /2
//     pad_bottom  = (640 - 720*0.5) - 70 = 210
struct Letterbox {
    float gain;
    float pad_top, pad_bottom, pad_left, pad_right;
};

static Letterbox compute_letterbox(const PluginCfg& c) {
    Letterbox lb{};
    // 1) Gain. Either an explicit override, or aspect-preserving fit.
    if (c.gain_override > 0.f) {
        lb.gain = c.gain_override;
    } else {
        lb.gain = min(static_cast<float>(c.model_h) / c.orig_h,
                      static_cast<float>(c.model_w) / c.orig_w);
    }

    // 2) Default pads. x is true-centered; y uses the SiMa preproc quirk
    //    (top = total_v_gap/4) -- see comment on PluginCfg::pad_top above.
    const float pad_y_total = static_cast<float>(c.model_h) - static_cast<float>(c.orig_h) * lb.gain;
    const float pad_x_total = static_cast<float>(c.model_w) - static_cast<float>(c.orig_w) * lb.gain;
    const float def_pad_top    = pad_y_total * 0.25f;            // SiMa: 1/4 on top
    const float def_pad_bottom = pad_y_total - def_pad_top;       // remainder on bottom
    const float def_pad_left   = pad_x_total * 0.5f;              // standard center
    const float def_pad_right  = pad_x_total - def_pad_left;

    // 3) Per-side overrides from JSON (any of pad_top / pad_bottom / pad_left
    //    / pad_right). -1 means "use computed default". Use these to pin
    //    values to a known-good calibration for a specific model.
    lb.pad_top    = (c.pad_top    >= 0.f) ? c.pad_top    : def_pad_top;
    lb.pad_bottom = (c.pad_bottom >= 0.f) ? c.pad_bottom : def_pad_bottom;
    lb.pad_left   = (c.pad_left   >= 0.f) ? c.pad_left   : def_pad_left;
    lb.pad_right  = (c.pad_right  >= 0.f) ? c.pad_right  : def_pad_right;
    return lb;
}

// Reverse letterbox: model-space cxcywh -> clamped frame-space xyxy.
// Uses pad_top / pad_left so asymmetric CENTER pad is handled correctly.
static inline void model_to_frame(const RawDet& d, const Letterbox& lb,
                                  int orig_w, int orig_h,
                                  float& x1, float& y1, float& x2, float& y2) {
    float fx1 = (d.cx - d.w * 0.5f - lb.pad_left) / lb.gain;
    float fy1 = (d.cy - d.h * 0.5f - lb.pad_top ) / lb.gain;
    float fx2 = (d.cx + d.w * 0.5f - lb.pad_left) / lb.gain;
    float fy2 = (d.cy + d.h * 0.5f - lb.pad_top ) / lb.gain;
    x1 = max(0.f, min(static_cast<float>(orig_w - 1), fx1));
    y1 = max(0.f, min(static_cast<float>(orig_h - 1), fy1));
    x2 = max(0.f, min(static_cast<float>(orig_w - 1), fx2));
    y2 = max(0.f, min(static_cast<float>(orig_h - 1), fy2));
}

// NCHW base pointer for tensor `t` (const float* into g_fp32_nchw).
static inline const float* tensor_nchw(int t) {
    return g_fp32_nchw.data() + g_tensor_off[t];
}

//==============================================================================
//  decode_type-specific candidate generators (NCHW indexing)
//==============================================================================
//
//  After the NHWC -> NCHW transpose, each tensor of shape [H, W, C] is laid
//  out as [C, H, W]. For element (c, y, x):
//      offset = c * (H*W) + y * W + x
//
//  Anchor-free detection uses two tensors per head:
//      bbox tensor (H, W, 4 or 64)  -> base_b
//      cls  tensor (H, W, num_cls)  -> base_c

// ---- yolov8 / v9 / v10 / v11 detection (anchor-free) -----------------------
static void decode_yolo_det(const PluginCfg& c, vector<RawDet>& out) {
    // When model emits logits we threshold in logit space (sigmoid is monotonic)
    // to skip an expf() call on rejected anchors. When the model already emits
    // probabilities (`class_is_prob`), we threshold directly.
    const float thr = c.class_is_prob
                    ? c.conf_thresh
                    : logf(c.conf_thresh / (1.f - c.conf_thresh));

    for (int hi = 0; hi < DETECTION_HEAD_COUNT; ++hi) {
        const int H        = c.heads_h[hi];
        const int W        = c.heads_w[hi];
        const int HW       = H * W;
        const int S        = c.strides[hi];
        const int bbox_ch  = c.tensor_d[hi];                                   // 4 or 64
        const float* bp    = tensor_nchw(hi);                                  // [bbox_ch, H, W]
        const float* cp    = tensor_nchw(DETECTION_HEAD_COUNT + hi);           // [num_cls, H, W]

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int hw = y * W + x;

                // Per-anchor argmax across class channels (NCHW: stride HW per channel).
                int   best_cls = 0;
                float best_v   = cp[hw];
                for (int k = 1; k < c.num_classes; ++k) {
                    const float v = cp[k * HW + hw];
                    if (v > best_v) { best_v = v; best_cls = k; }
                }
                if (best_v < thr) continue;
                const float conf = c.class_is_prob ? best_v : sigmoid_f(best_v);

                // bbox: 4 or 64 channels at NCHW stride HW. Three decode paths:
                //
                // 1) DFL bins (depth=64):       [bins_l(16), bins_t(16), bins_r(16), bins_b(16)]
                //                               -> softmax-EV per side -> distances in grid units
                //                               -> add anchor center, multiply by stride.
                //
                // 2) cxcywh_pixel (depth=4, default for SiMa surgeon_yolov8):
                //                               4 channels are [cx, cy, w, h] already in
                //                               model-space pixels. Anchor offset and stride
                //                               multiplication are baked into the model graph.
                //                               Read them as-is.
                //
                // 3) ltrb_grid (depth=4, opt-in):
                //                               4 channels are raw [l, t, r, b] distances in
                //                               grid units. Apply anchor + stride in C++.
                RawDet d;
                if (c.bbox_is_dfl && bbox_ch >= DFL_BBOX_CH) {
                    float bins[DFL_BBOX_CH];
                    for (int k = 0; k < DFL_BBOX_CH; ++k)
                        bins[k] = bp[k * HW + hw];
                    const float dl = dfl_decode(bins +  0);
                    const float dt = dfl_decode(bins + 16);
                    const float dr = dfl_decode(bins + 32);
                    const float db = dfl_decode(bins + 48);
                    const float acx = (x + 0.5f) * S;
                    const float acy = (y + 0.5f) * S;
                    d.cx = acx + (dr - dl) * 0.5f * S;
                    d.cy = acy + (db - dt) * 0.5f * S;
                    d.w  = (dl + dr) * S;
                    d.h  = (dt + db) * S;
                } else if (c.bbox_format == "ltrb_grid") {
                    const float dl = bp[0 * HW + hw];
                    const float dt = bp[1 * HW + hw];
                    const float dr = bp[2 * HW + hw];
                    const float db = bp[3 * HW + hw];
                    const float acx = (x + 0.5f) * S;
                    const float acy = (y + 0.5f) * S;
                    d.cx = acx + (dr - dl) * 0.5f * S;
                    d.cy = acy + (db - dt) * 0.5f * S;
                    d.w  = (dl + dr) * S;
                    d.h  = (dt + db) * S;
                } else {
                    // "cxcywh_pixel" -- SiMa surgeon_yolov8 active path.
                    d.cx = bp[0 * HW + hw];
                    d.cy = bp[1 * HW + hw];
                    d.w  = bp[2 * HW + hw];
                    d.h  = bp[3 * HW + hw];
                }
                d.score  = conf;
                d.label  = best_cls;
                d.head   = hi;
                d.y_cell = y;
                d.x_cell = x;
                out.push_back(d);
            }
        }
    }
}

// ---- yolov5 / yolov7 anchor-based detection --------------------------------
//   Single tensor per head, channel layout (post NHWC->NCHW): [na*(5+num_cls), H, W]
//   For anchor `ai` and field `f` in [0, 5+num_classes):
//       channel = ai * (5 + num_cls) + f
static void decode_yolo_anchor(const PluginCfg& c, vector<RawDet>& out) {
    const int na    = c.num_anchors;
    const int per_a = 5 + c.num_classes;

    for (int hi = 0; hi < DETECTION_HEAD_COUNT; ++hi) {
        const int H  = c.heads_h[hi];
        const int W  = c.heads_w[hi];
        const int HW = H * W;
        const int S  = c.strides[hi];
        const float* tp = tensor_nchw(hi);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int hw = y * W + x;
                for (int ai = 0; ai < na; ++ai) {
                    const float* d = tp + ai * per_a * HW;        // base for anchor `ai`
                    const float obj = sigmoid_f(d[4 * HW + hw]);
                    if (obj < c.conf_thresh) continue;

                    int   best_cls = 0;
                    float best_v   = d[5 * HW + hw];
                    for (int k = 1; k < c.num_classes; ++k) {
                        const float v = d[(5 + k) * HW + hw];
                        if (v > best_v) { best_v = v; best_cls = k; }
                    }
                    const float score = obj * sigmoid_f(best_v);
                    if (score < c.conf_thresh) continue;

                    const float aw = c.anchors_wh[hi][ai][0];
                    const float ah = c.anchors_wh[hi][ai][1];
                    const float sx = sigmoid_f(d[0 * HW + hw]) * 2.f - 0.5f;
                    const float sy = sigmoid_f(d[1 * HW + hw]) * 2.f - 0.5f;
                    const float sw = sigmoid_f(d[2 * HW + hw]) * 2.f;
                    const float sh = sigmoid_f(d[3 * HW + hw]) * 2.f;

                    RawDet rd;
                    rd.cx     = (sx + x) * S;
                    rd.cy     = (sy + y) * S;
                    rd.w      = sw * sw * aw;
                    rd.h      = sh * sh * ah;
                    rd.score  = score;
                    rd.label  = best_cls;
                    rd.head   = hi;
                    rd.y_cell = y;
                    rd.x_cell = x;
                    out.push_back(rd);
                }
            }
        }
    }
}

// ---- yolov5-seg / yolov7-seg ----------------------------------------------
static void decode_yolo_anchor_seg(const PluginCfg& c, vector<RawDet>& out) {
    const int na    = c.num_anchors;
    const int per_a = 5 + c.num_classes + MASK_PROTO_CH;

    for (int hi = 0; hi < DETECTION_HEAD_COUNT; ++hi) {
        const int H  = c.heads_h[hi];
        const int W  = c.heads_w[hi];
        const int HW = H * W;
        const int S  = c.strides[hi];
        const float* tp = tensor_nchw(hi);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int hw = y * W + x;
                for (int ai = 0; ai < na; ++ai) {
                    const float* d = tp + ai * per_a * HW;
                    const float obj = sigmoid_f(d[4 * HW + hw]);
                    if (obj < c.conf_thresh) continue;

                    int   best_cls = 0;
                    float best_v   = d[5 * HW + hw];
                    for (int k = 1; k < c.num_classes; ++k) {
                        const float v = d[(5 + k) * HW + hw];
                        if (v > best_v) { best_v = v; best_cls = k; }
                    }
                    const float score = obj * sigmoid_f(best_v);
                    if (score < c.conf_thresh) continue;

                    const float aw = c.anchors_wh[hi][ai][0];
                    const float ah = c.anchors_wh[hi][ai][1];
                    const float sx = sigmoid_f(d[0 * HW + hw]) * 2.f - 0.5f;
                    const float sy = sigmoid_f(d[1 * HW + hw]) * 2.f - 0.5f;
                    const float sw = sigmoid_f(d[2 * HW + hw]) * 2.f;
                    const float sh = sigmoid_f(d[3 * HW + hw]) * 2.f;

                    RawDet rd;
                    rd.cx     = (sx + x) * S;
                    rd.cy     = (sy + y) * S;
                    rd.w      = sw * sw * aw;
                    rd.h      = sh * sh * ah;
                    rd.score  = score;
                    rd.label  = best_cls;
                    rd.head   = hi;
                    rd.y_cell = y;
                    rd.x_cell = x;
                    for (int i = 0; i < MASK_PROTO_CH; ++i)
                        rd.mcoef[i] = d[(5 + c.num_classes + i) * HW + hw];
                    out.push_back(rd);
                }
            }
        }
    }
}

// ---- yolox detection -------------------------------------------------------
// Channel layout per cell: [tx, ty, tw, th, obj, cls_0, ..., cls_{nc-1}].
// Two export styles in the wild:
//   class_is_prob = false (default for raw export): obj/cls are LOGITS
//                   -> sigmoid both before multiplying.
//   class_is_prob = true                          : obj/cls already sigmoided
//                   (typical when the model graph bakes sigmoid into the head)
//                   -> use raw values directly.
//
// We argmax on raw class values first (sigmoid is monotonic so argmax is
// preserved) and only sigmoid the chosen class -- saves an exp() per anchor
// per class on rejected candidates.
static void decode_yolox(const PluginCfg& c, vector<RawDet>& out) {
    for (int hi = 0; hi < DETECTION_HEAD_COUNT; ++hi) {
        const int H  = c.heads_h[hi];
        const int W  = c.heads_w[hi];
        const int HW = H * W;
        const int S  = c.strides[hi];
        const float* tp = tensor_nchw(hi);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int hw   = y * W + x;
                const float obj_raw = tp[4 * HW + hw];

                // argmax over class channels on raw values (sigmoid-monotonic).
                int   best_cls   = 0;
                float best_logit = tp[5 * HW + hw];
                for (int k = 1; k < c.num_classes; ++k) {
                    const float v = tp[(5 + k) * HW + hw];
                    if (v > best_logit) { best_logit = v; best_cls = k; }
                }

                const float obj = c.class_is_prob ? obj_raw   : sigmoid_f(obj_raw);
                const float cls = c.class_is_prob ? best_logit: sigmoid_f(best_logit);
                const float score = obj * cls;
                if (score <= c.conf_thresh) continue;

                RawDet rd;
                rd.cx     = (tp[0 * HW + hw] + x) * S;
                rd.cy     = (tp[1 * HW + hw] + y) * S;
                rd.w      = expf(tp[2 * HW + hw]) * S;
                rd.h      = expf(tp[3 * HW + hw]) * S;
                rd.score  = score;
                rd.label  = best_cls;
                rd.head   = hi;
                rd.y_cell = y;
                rd.x_cell = x;
                out.push_back(rd);
            }
        }
    }
}

// ---- yolov8 / yolov11 pose -------------------------------------------------
static void decode_yolo_pose(const PluginCfg& c, vector<RawDet>& out) {
    decode_yolo_det(c, out);
}

static void compute_keypoints(const PluginCfg& c,
                              const RawDet& det, const Letterbox& lb,
                              PosePoint* dst /* size = num_keypoints */) {
    const int kp_t = 2 * DETECTION_HEAD_COUNT + det.head;
    const float* kp = tensor_nchw(kp_t);
    const int   H   = c.tensor_h[kp_t];
    const int   W   = c.tensor_w[kp_t];
    const int   HW  = H * W;
    const int   S   = c.strides[det.head];
    const float anchor_x = det.x_cell + 0.5f;
    const float anchor_y = det.y_cell + 0.5f;
    const float inv_gain = 1.f / lb.gain;
    const float fwm = static_cast<float>(c.orig_w - 1);
    const float fhm = static_cast<float>(c.orig_h - 1);
    const int   hw  = det.y_cell * W + det.x_cell;

    for (int k = 0; k < c.num_keypoints; ++k) {
        const float kx_reg = kp[(k * 3 + 0) * HW + hw];
        const float ky_reg = kp[(k * 3 + 1) * HW + hw];
        const float kc_raw = kp[(k * 3 + 2) * HW + hw];

        // Ultralytics-style pose decode in model space.
        float mx = (kx_reg * 2.f - 0.5f + anchor_x) * S;
        float my = (ky_reg * 2.f - 0.5f + anchor_y) * S;

        float fx = (mx - lb.pad_left) * inv_gain;
        float fy = (my - lb.pad_top ) * inv_gain;
        fx = max(0.f, min(fwm, fx));
        fy = max(0.f, min(fhm, fy));

        dst[k]._x       = static_cast<uint32_t>(fx);
        dst[k]._y       = static_cast<uint32_t>(fy);
        dst[k]._visible = sigmoid_f(kc_raw);
    }
}

// ---- yolov8 / v9 / v10 / v11 seg -------------------------------------------
static void decode_yolo_seg(const PluginCfg& c, vector<RawDet>& out) {
    decode_yolo_det(c, out);

    for (auto& d : out) {
        const int    mc_t = 2 * DETECTION_HEAD_COUNT + d.head;
        const float* mc   = tensor_nchw(mc_t);
        const int    H    = c.tensor_h[mc_t];
        const int    W    = c.tensor_w[mc_t];
        const int    HW   = H * W;
        const int    Cd   = c.tensor_d[mc_t];
        const int    n    = min<int>(Cd, MASK_PROTO_CH);
        const int    hw   = d.y_cell * W + d.x_cell;
        for (int i = 0; i < n; ++i) d.mcoef[i] = mc[i * HW + hw];
    }
}

//==============================================================================
//  Output writers
//==============================================================================
static inline void fill_bbox_out(BoundingBoxOut& dst, const RawDet& d,
                                 const Letterbox& lb, int orig_w, int orig_h) {
    float x1, y1, x2, y2;
    model_to_frame(d, lb, orig_w, orig_h, x1, y1, x2, y2);
    dst._x        = static_cast<uint32_t>(x1);
    dst._y        = static_cast<uint32_t>(y1);
    dst._w        = static_cast<uint32_t>(max(0.f, x2 - x1));
    dst._h        = static_cast<uint32_t>(max(0.f, y2 - y1));
    dst._score    = d.score;
    dst._class_id = static_cast<uint32_t>(d.label);
}

// BBox: [int num_det][topK x BoundingBoxOut]
//   Render's RENDER_TYPE::BOX reads `num_det` from the first 4 bytes, then
//   iterates that many `Points` (24B each). We MUST guarantee num_det is in
//   [0, topK] so the renderer never walks off the end of our 580-byte buffer.
static void write_bbox_output(span<uint8_t> out, const vector<RawDet>& dets,
                              const PluginCfg& c, const Letterbox& lb) {
    if (out.size() < sizeof(int)) {
        cerr << "[yolodecode] BBox output buffer < 4 bytes (" << out.size() << ")" << endl;
        return;
    }
    // Caller already zeroed `out`; first 4 bytes are 0 -> num_det = 0 (safe).
    const size_t need = sizeof(int) + size_t(c.topk) * sizeof(BoundingBoxOut);
    if (out.size() < need) {
        cerr << "[yolodecode] BBox output too small: " << out.size()
             << " < " << need << " (topK=" << c.topk
             << "). Writing num_det=0 only." << endl;
        return;
    }
    int n = static_cast<int>(min<size_t>(dets.size(), c.topk));
    *reinterpret_cast<int*>(out.data()) = n;
    BoundingBoxOut* bb = reinterpret_cast<BoundingBoxOut*>(out.data() + sizeof(int));
    for (int i = 0; i < n; ++i) fill_bbox_out(bb[i], dets[i], lb, c.orig_w, c.orig_h);
}

// Pose: [int num_det][topK x BoundingBoxOut][topK x (kp x PosePoint)]
static void write_pose_output(span<uint8_t> out, vector<RawDet>& dets,
                              const PluginCfg& c, const Letterbox& lb) {
    const size_t bbox_block = size_t(c.topk) * sizeof(BoundingBoxOut);
    const size_t pose_block = size_t(c.topk) * size_t(c.num_keypoints) * sizeof(PosePoint);
    const size_t need = sizeof(int) + bbox_block + pose_block;
    if (out.size() < need) {
        cerr << "[yolodecode] Pose output too small: " << out.size()
             << " < " << need << endl;
        return;
    }
    memset(out.data(), 0, need);
    int n = static_cast<int>(min<size_t>(dets.size(), c.topk));
    *reinterpret_cast<int*>(out.data()) = n;

    BoundingBoxOut* bb = reinterpret_cast<BoundingBoxOut*>(out.data() + sizeof(int));
    PosePoint*      pp = reinterpret_cast<PosePoint*>(out.data() + sizeof(int) + bbox_block);

    for (int i = 0; i < n; ++i) {
        fill_bbox_out(bb[i], dets[i], lb, c.orig_w, c.orig_h);
        compute_keypoints(c, dets[i], lb, pp + size_t(i) * c.num_keypoints);
    }
}

// Render one binary mask (sigmoid(proto . coef) > 0.5 -> 255 else 0) into 160x160.
// `proto` is the prototype tensor in NCHW: [MASK_PROTO_CH, MASK_H, MASK_W].
static void render_instance_mask(const float* proto,
                                 const array<float, MASK_PROTO_CH>& coef,
                                 const RawDet& d,
                                 uint8_t* tile /* 160*160 */) {
    const float sx = static_cast<float>(MASK_W) / static_cast<float>(g_cfg.model_w);
    const float sy = static_cast<float>(MASK_H) / static_cast<float>(g_cfg.model_h);
    int x1 = max(0,         static_cast<int>((d.cx - d.w * 0.5f) * sx));
    int y1 = max(0,         static_cast<int>((d.cy - d.h * 0.5f) * sy));
    int x2 = min(MASK_W - 1, static_cast<int>((d.cx + d.w * 0.5f) * sx));
    int y2 = min(MASK_H - 1, static_cast<int>((d.cy + d.h * 0.5f) * sy));

    memset(tile, 0, MASK_PIXELS);
    if (x2 < x1 || y2 < y1) return;

    const int HW = MASK_H * MASK_W;
    for (int y = y1; y <= y2; ++y) {
        uint8_t* trow = tile + y * MASK_W;
        for (int x = x1; x <= x2; ++x) {
            const int hw = y * MASK_W + x;
            float acc = 0.f;
            for (int cc = 0; cc < MASK_PROTO_CH; ++cc) {
                acc += proto[cc * HW + hw] * coef[cc];
            }
            trow[x] = (sigmoid_f(acc) > 0.5f) ? 255 : 0;
        }
    }
}

// Seg: [int num_det][topK x (BoundingBoxOut + uint8 mask[160*160])]
static void write_seg_output(span<uint8_t> out, const vector<RawDet>& dets,
                             const PluginCfg& c, const Letterbox& lb) {
    const size_t per_inst = sizeof(BoundingBoxOut) + MASK_PIXELS;
    const size_t need     = sizeof(int) + size_t(c.topk) * per_inst;
    if (out.size() < need) {
        cerr << "[yolodecode] Seg output too small: " << out.size()
             << " < " << need << endl;
        return;
    }
    memset(out.data(), 0, need);
    int n = static_cast<int>(min<size_t>(dets.size(), c.topk));
    *reinterpret_cast<int*>(out.data()) = n;

    const int    proto_idx = c.num_in_tensor - 1;
    const float* proto     = tensor_nchw(proto_idx);

    uint8_t* slot = out.data() + sizeof(int);
    for (int i = 0; i < n; ++i) {
        BoundingBoxOut* bb   = reinterpret_cast<BoundingBoxOut*>(slot);
        uint8_t*        tile = slot + sizeof(BoundingBoxOut);
        fill_bbox_out(*bb, dets[i], lb, c.orig_w, c.orig_h);
        render_instance_mask(proto, dets[i].mcoef, dets[i], tile);
        slot += per_inst;
    }
}

//==============================================================================
//  UserContext
//==============================================================================
UserContext::UserContext(GstObject* /*obj*/, const char* json_file_name) {
    try {
        ifstream f(json_file_name);
        if (!f.is_open()) {
            cerr << "[yolodecode] Cannot open JSON: " << json_file_name << endl;
            return;
        }
        json cfg;
        f >> cfg;

        if (cfg.contains("decode_type"))         g_cfg.decode_type_str = cfg["decode_type"].get<string>();
        if (cfg.contains("render_type"))         g_cfg.render_type_str = cfg["render_type"].get<string>();
        if (cfg.contains("num_classes"))         g_cfg.num_classes     = cfg["num_classes"].get<int>();
        if (cfg.contains("num_keypoints"))       g_cfg.num_keypoints   = cfg["num_keypoints"].get<int>();
        if (cfg.contains("model_width"))         g_cfg.model_w         = cfg["model_width"].get<int>();
        if (cfg.contains("model_height"))        g_cfg.model_h         = cfg["model_height"].get<int>();
        if (cfg.contains("original_width"))      g_cfg.orig_w          = cfg["original_width"].get<int>();
        if (cfg.contains("original_height"))     g_cfg.orig_h          = cfg["original_height"].get<int>();
        if (cfg.contains("topk"))                g_cfg.topk            = cfg["topk"].get<int>();
        if (cfg.contains("detection_threshold")) g_cfg.conf_thresh     = cfg["detection_threshold"].get<float>();
        if (cfg.contains("nms_iou_threshold"))   g_cfg.nms_thresh      = cfg["nms_iou_threshold"].get<float>();
        if (cfg.contains("num_in_tensor"))       g_cfg.num_in_tensor   = cfg["num_in_tensor"].get<int>();
        if (cfg.contains("input_depth"))         g_cfg.tensor_d        = cfg["input_depth"].get<vector<int>>();
        if (cfg.contains("input_height"))        g_cfg.tensor_h        = cfg["input_height"].get<vector<int>>();
        if (cfg.contains("input_width"))         g_cfg.tensor_w        = cfg["input_width"].get<vector<int>>();
        if (cfg.contains("class_is_prob"))       g_cfg.class_is_prob   = cfg["class_is_prob"].get<bool>();
        if (cfg.contains("bbox_format"))         g_cfg.bbox_format     = cfg["bbox_format"].get<string>();
        // Inverse-letterbox overrides. Any subset can be set; missing values
        // fall back to compute_letterbox()'s defaults. For 1280x720 -> 640x640
        // the defaults give gain=0.5, pad_left=0, pad_right=0, pad_top=70,
        // pad_bottom=210 (SiMa preproc places only 1/4 of the vertical gap on
        // top -- see comment on PluginCfg::pad_top). Override here if you
        // calibrated against a different preproc.
        if (cfg.contains("pad_top"))             g_cfg.pad_top         = cfg["pad_top"].get<float>();
        if (cfg.contains("pad_bottom"))          g_cfg.pad_bottom      = cfg["pad_bottom"].get<float>();
        if (cfg.contains("pad_left"))            g_cfg.pad_left        = cfg["pad_left"].get<float>();
        if (cfg.contains("pad_right"))           g_cfg.pad_right       = cfg["pad_right"].get<float>();
        if (cfg.contains("letterbox_gain"))      g_cfg.gain_override   = cfg["letterbox_gain"].get<float>();
        if (cfg.contains("buffers") && cfg["buffers"].contains("output"))
            g_cfg.output_size = cfg["buffers"]["output"]["size"].get<size_t>();
        if (cfg.contains("system") && cfg["system"].contains("debug"))
            g_cfg.debug_log = cfg["system"]["debug"].get<int>() != 0;

        // Optional anchor override for yolov5/v7. Shape:
        //   "anchors": [ [[w,h],[w,h],[w,h]],   <- P3/8
        //                [[w,h],[w,h],[w,h]],   <- P4/16
        //                [[w,h],[w,h],[w,h]] ]  <- P5/32
        if (cfg.contains("anchors") && cfg["anchors"].is_array()) {
            const auto& a = cfg["anchors"];
            for (int hi = 0; hi < DETECTION_HEAD_COUNT && hi < (int)a.size(); ++hi) {
                const auto& row = a[hi];
                for (int ai = 0; ai < g_cfg.num_anchors && ai < (int)row.size(); ++ai) {
                    g_cfg.anchors_wh[hi][ai][0] = row[ai][0].get<float>();
                    g_cfg.anchors_wh[hi][ai][1] = row[ai][1].get<float>();
                }
            }
        }
        if (cfg.contains("num_anchors")) g_cfg.num_anchors = cfg["num_anchors"].get<int>();

        g_cfg.kind = kind_from_string(g_cfg.decode_type_str);
        if (g_cfg.kind == DecodeKind::UNKNOWN) {
            cerr << "[yolodecode] decode_type='" << g_cfg.decode_type_str
                 << "' is not recognised. Set one of:\n"
                 << "  Detection (anchor-free, 6 tensors):  yolov8 / yolov9 / yolov10 / "
                    "yolov11 / yolov12 / yolo26\n"
                 << "  Detection (anchor-based, 3 tensors): yolov5 / yolov7\n"
                 << "  Detection (YOLOX, 3 tensors):        yolox\n"
                 << "  Pose (9 tensors):                    yolov8-pose / yolov11-pose\n"
                 << "  Segmentation (10 tensors):           yolov8-seg / yolov9-seg / "
                    "yolov10-seg / yolov11-seg\n"
                 << "  Segmentation (4 tensors, anchor):    yolov5-seg / yolov7-seg\n"
                 << "  See cfg/MODELS.md for per-model JSON templates." << endl;
            return;
        }

        // Sanity check: tensor count vs decode kind.
        const int expected_tensors =
              (g_cfg.kind == DecodeKind::YOLOX_DET)        ? 3
            : (g_cfg.kind == DecodeKind::YOLO_DET_ANCHOR)  ? 3
            : (g_cfg.kind == DecodeKind::YOLO_DET)         ? 6
            : (g_cfg.kind == DecodeKind::YOLO_POSE)        ? 9
            : (g_cfg.kind == DecodeKind::YOLO_SEG)         ? 10
            : (g_cfg.kind == DecodeKind::YOLO_SEG_ANCHOR)  ? 4
            : 0;
        if (expected_tensors > 0 && g_cfg.num_in_tensor != expected_tensors) {
            cerr << "[yolodecode] WARNING: decode_type=" << g_cfg.decode_type_str
                 << " expects " << expected_tensors << " tensors, JSON has "
                 << g_cfg.num_in_tensor << endl;
        }
        if (static_cast<int>(g_cfg.tensor_d.size()) < g_cfg.num_in_tensor ||
            static_cast<int>(g_cfg.tensor_h.size()) < g_cfg.num_in_tensor ||
            static_cast<int>(g_cfg.tensor_w.size()) < g_cfg.num_in_tensor) {
            cerr << "[yolodecode] input_depth/height/width arrays shorter than num_in_tensor" << endl;
            return;
        }

        for (int hi = 0; hi < DETECTION_HEAD_COUNT; ++hi) {
            g_cfg.heads_h[hi] = g_cfg.tensor_h[hi];
            g_cfg.heads_w[hi] = g_cfg.tensor_w[hi];
            g_cfg.strides[hi] = (g_cfg.heads_h[hi] > 0)
                                ? (g_cfg.model_h / g_cfg.heads_h[hi])
                                : DEFAULT_STRIDES[hi];
        }
        if (g_cfg.kind == DecodeKind::YOLO_DET ||
            g_cfg.kind == DecodeKind::YOLO_POSE ||
            g_cfg.kind == DecodeKind::YOLO_SEG) {
            g_cfg.bbox_is_dfl = (g_cfg.tensor_d[0] >= DFL_BBOX_CH);
        }

        size_t total = 0;
        for (int i = 0; i < g_cfg.num_in_tensor; ++i) total += tensor_size(g_cfg, i);
        g_fp32_nhwc.resize(total);
        g_fp32_nchw.resize(total);
        g_tensor_off.assign(g_cfg.num_in_tensor, 0);

        const Letterbox lb_dbg = compute_letterbox(g_cfg);
        cout << "[yolodecode] decode=" << g_cfg.decode_type_str
             << " render=" << g_cfg.render_type_str
             << " classes=" << g_cfg.num_classes
             << " kpts=" << g_cfg.num_keypoints
             << " model=" << g_cfg.model_w << "x" << g_cfg.model_h
             << " frame=" << g_cfg.orig_w << "x" << g_cfg.orig_h
             << " topk=" << g_cfg.topk
             << " conf=" << g_cfg.conf_thresh
             << " nms="  << g_cfg.nms_thresh
             << " bbox_dfl=" << g_cfg.bbox_is_dfl
             << " bbox_format=" << g_cfg.bbox_format
             << " class_is_prob=" << g_cfg.class_is_prob
             << " strides=[" << g_cfg.strides[0] << "," << g_cfg.strides[1] << "," << g_cfg.strides[2] << "]"
             << " letterbox=[gain=" << lb_dbg.gain
                            << " pad_top=" << lb_dbg.pad_top
                            << " pad_bottom=" << lb_dbg.pad_bottom
                            << " pad_left=" << lb_dbg.pad_left
                            << " pad_right=" << lb_dbg.pad_right << "]"
             << " out_buf=" << g_cfg.output_size << endl;

        g_cfg.initialized = true;
    } catch (const exception& e) {
        cerr << "[yolodecode] JSON error: " << e.what() << endl;
    }
}

UserContext::~UserContext() {}

//==============================================================================
//  run() - per-frame entry point
//==============================================================================
void UserContext::run(vector<Input>& input, span<uint8_t> output) {
    if (!g_cfg.initialized || input.empty()) {
        if (!output.empty()) memset(output.data(), 0, output.size());
        return;
    }

    // -- 1. BF16 (16-bit) -> FP32 (32-bit). Layout still NHWC, all tensors
    //       concatenated back-to-back. -----------------------------------------
    const uint16_t* src    = reinterpret_cast<const uint16_t*>(input[0].getData().data());
    const int       n_elem = input[0].getDataSize() / 2;

    size_t total_elem = 0;
    for (int t = 0; t < g_cfg.num_in_tensor; ++t)
        total_elem += static_cast<size_t>(tensor_size(g_cfg, t));
    if (g_fp32_nhwc.size() < total_elem) g_fp32_nhwc.resize(total_elem);
    if (g_fp32_nchw.size() < total_elem) g_fp32_nchw.resize(total_elem);

    bf16_to_fp32(src, g_fp32_nhwc.data(), min<int>(n_elem, static_cast<int>(total_elem)));

    // -- 2. NHWC -> NCHW transpose for every tensor. After this, each tensor
    //       lives at g_fp32_nchw + g_tensor_off[t] in [C, H, W] layout. -------
    transpose_all_to_nchw(g_cfg);

    // -- 3. Generate raw candidates ------------------------------------------
    vector<RawDet> dets;
    dets.reserve(2048);

    switch (g_cfg.kind) {
        case DecodeKind::YOLO_DET:        decode_yolo_det        (g_cfg, dets); break;
        case DecodeKind::YOLO_DET_ANCHOR: decode_yolo_anchor     (g_cfg, dets); break;
        case DecodeKind::YOLOX_DET:       decode_yolox           (g_cfg, dets); break;
        case DecodeKind::YOLO_POSE:       decode_yolo_pose       (g_cfg, dets); break;
        case DecodeKind::YOLO_SEG:        decode_yolo_seg        (g_cfg, dets); break;
        case DecodeKind::YOLO_SEG_ANCHOR: decode_yolo_anchor_seg (g_cfg, dets); break;
        default:
            memset(output.data(), 0, output.size());
            return;
    }

    const size_t pre_nms = dets.size();

    // -- 4. NMS + topK -------------------------------------------------------
    per_class_nms(dets, g_cfg.nms_thresh);
    if (static_cast<int>(dets.size()) > g_cfg.topk) dets.resize(g_cfg.topk);

    // -- 5. Write output (4 + topK*24 for BBox -- e.g. 580 B at topk=24) -----
    // Defensive: always zero the entire output buffer first so a downstream
    // reader can never see stale bytes if a writer below early-returns.
    if (!output.empty()) memset(output.data(), 0, output.size());

    const Letterbox lb = compute_letterbox(g_cfg);

    switch (g_cfg.kind) {
        case DecodeKind::YOLO_DET:
        case DecodeKind::YOLO_DET_ANCHOR:
        case DecodeKind::YOLOX_DET:
            write_bbox_output(output, dets, g_cfg, lb);
            break;
        case DecodeKind::YOLO_POSE:
            write_pose_output(output, dets, g_cfg, lb);
            break;
        case DecodeKind::YOLO_SEG:
        case DecodeKind::YOLO_SEG_ANCHOR:
            write_seg_output (output, dets, g_cfg, lb);
            break;
        default:
            break;        // already zeroed above
    }

    // Always log the first 3 frames (regardless of debug flag) so a downstream
    // segfault is easy to diagnose. After that, gate on `system.debug`.
    static int s_frames = 0;
    const bool log_now = g_cfg.debug_log || (s_frames < 3);
    if (log_now) {
        const int n_written = std::min<int>(static_cast<int>(dets.size()), g_cfg.topk);
        cout << "[yolodecode] frame=" << s_frames
             << " in_bytes=" << input[0].getDataSize()
             << " out_bytes=" << output.size()
             << " " << g_cfg.decode_type_str
             << " pre_nms=" << pre_nms
             << " dets=" << dets.size() << "/" << g_cfg.topk
             << " written=" << n_written;
        for (int i = 0; i < n_written && i < 8; ++i) {
            float x1, y1, x2, y2;
            model_to_frame(dets[i], lb, g_cfg.orig_w, g_cfg.orig_h, x1, y1, x2, y2);
            cout << "  [#" << i << " cls=" << dets[i].label
                 << " s=" << dets[i].score
                 << " xywh=" << (int)x1 << "," << (int)y1
                 << "," << (int)(x2-x1) << "," << (int)(y2-y1) << "]";
        }
        cout << endl;
    }
    ++s_frames;
}
