# FocalLoss
Caffe implementation of FAIR paper "Focal Loss for Dense Object Detection" for SSD.
```
layer {
  name: "mbox_loss"
  type: "MultiBoxFocalLoss" #change the type
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  loss_param {
    normalization: VALID
  }
  focal_loss_param { #set the alpha and gamma, default is alpha=0.25, gamma=2.0
    alpha: 0.25
    gamma: 2.0
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1.0
    num_classes: 21
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.5
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.5
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: NONE #do not use OHEM
  }
}
```
