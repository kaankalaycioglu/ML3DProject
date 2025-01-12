import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate, AnchorHeadTemplateERPN


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class AnchorHeadSingleERPN(AnchorHeadTemplateERPN):
    def __init__(
        self,
        model_cfg,
        input_channels,
        num_class,
        class_names,
        grid_size,
        point_cloud_range,
        predict_boxes_when_training=True,
        **kwargs
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training,
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # self.dfpn3_num_features = 512
        # self.dfpn5_num_features = 512

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_channels, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),
        # )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),
        # )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True)
        # )

        # self.conv5_dfpn = nn.Sequential(
        #     nn.ConvTranspose2d(512, self.dfpn5_num_features),
        #     nn.Conv2d(self.dfpn3_num_features, self.dfpn5_num_features, 3, padding=1),
        #     nn.BatchNorm2d(self.dfpn5_num_features),
        #     nn.ReLU(),
        # )

        # self.conv3_dfpn = nn.Sequential(
        #     nn.ConvTranspose2d(self.dfpn5_num_features+256, self.dfpn3_num_features),
        #     nn.Conv2d(self.dfpn3_num_features, self.dfpn3_num_features, 3, padding=1),
        #     nn.BatchNorm2d(self.dfpn3_num_features),
        #     nn.ReLU(),
        # )

        # self.max_pool = nn.MaxPool2d()
        # self.avg_pool = nn.AvgPool2d()
        # self.bn256 = nn.BatchNorm2d(256)
        # self.bn128 = nn.BatchNorm2d(128)

        self.auxiliary_conv_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.num_class,
            kernel_size=1,
        )
        self.auxiliary_conv_box = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1,
        )

        self.auxiliary_conv_combined = nn.Conv2d(
            self.num_anchors_per_location * (self.num_class + self.box_coder.code_size),
            input_channels,
            kernel_size=1,
        )

        self.main_conv_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.num_class,
            kernel_size=1,
        )

        self.main_conv_box = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1,
        )

        if self.model_cfg.get("USE_DIRECTION_CLASSIFIER", None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        nn.init.constant_(self.auxiliary_conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.auxiliary_conv_box.weight, mean=0, std=0.001)
        nn.init.constant_(self.main_conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.main_conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict["spatial_features_2d"]

        # out_2 = self.conv2(self.conv1(spatial_features_2d))
        # out_3 = self.conv3(out_2)
        # out_5 = self.conv5(self.conv4(out_3))

        # d_out_5 = self.conv5_dfpn(out_5)
        # out_3 = self.bn256(out_3)
        # d_out_3 = self.conv3_dfpn(torch.cat((out_3, d_out_5), dim=1))
        # out_2 = self.bn128(out_2)
        # d_out_2 = torch.cat((out_2, d_out_3), dim=1)
        # out_max = self.max_pool(d_out_2)
        # out_avg = self.avg_pool(d_out_2)
        # out = torch.cat((out_max, out_avg), dim=1)

        aux_cls_preds = self.auxiliary_conv_cls(spatial_features_2d)
        aux_box_preds = self.auxiliary_conv_box(spatial_features_2d)

        combined_aux_preds = torch.cat((aux_cls_preds, aux_box_preds), dim=1)

        conv_aux_preds = self.auxiliary_conv_combined(combined_aux_preds)

        main_preds_input = spatial_features_2d + conv_aux_preds

        main_cls_preds = self.main_conv_cls(main_preds_input)
        main_box_preds = self.main_conv_box(main_preds_input)

        main_cls_preds = main_cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        main_box_preds = main_box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict["cls_preds"] = main_cls_preds
        self.forward_ret_dict["box_preds"] = main_box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict["dir_cls_preds"] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict["gt_boxes"])
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict["batch_size"],
                cls_preds=main_cls_preds,
                box_preds=main_box_preds,
                dir_cls_preds=dir_cls_preds,
            )
            data_dict["batch_cls_preds"] = batch_cls_preds
            data_dict["batch_box_preds"] = batch_box_preds
            data_dict["cls_preds_normalized"] = False

        return data_dict