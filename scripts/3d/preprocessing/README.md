# 数据预处理

保留原 VEGA-3D 的场景数据与任务标注转换。当前已实际检查的是 ScanQA 小规模数据；其余转换脚本未在本次 Qwen3.5/Wan2.2 环境重新执行。

| 输入 | 脚本 | 结果 |
|---|---|---|
| ScanNet `.sens` | `generate_image_scannet.py` | RGB、深度、位姿、相机内参 |
| ScanNet 场景点云与标注 | `extract_scannet_pcd.py` | 场景点云和物体框 |
| 训练真值 / 验证检测框 | `extract_gt_box.py` / `extract_pred_box.py` | `data/metadata/` 中的框标注 |
| 场景点云 | `convert_pcd_to_voxel.py`、`max_coverage_sampling.py` | 覆盖采样所用的体素与帧列表 |
| 各任务原始标注 | `process_scanqa.py`、`process_sqa3d.py`、`process_scanrefer.py`、`process_multi3drefer.py`、`process_scan2cap.py` | `data/processed/*_llava_style.json` |

转换脚本按原目录读取 `data/scannet`、`data/embodiedscan`、`data/metadata` 及对应任务标注。使用前查看各脚本的参数和输入路径；点云处理可能另需 Open3D、plyfile 等依赖，当前环境清单不表示这些预处理程序均已验证。

问答数据中的 `human` 和 `gpt` 是输入、回答的角色字段。回答沿用数据集标注，字段名不表示调用 GPT 生成数据。

当前 foresight 小规模清单由 `scripts/3d/preprocessing/prepare_foresight_data.py` 生成，详细配置见 `docs/foresight.md`。Wan2.2 prompt 导出使用本目录的 `export_wan22_prompt_embedding.py`；离线特征读取和验证场景补齐入口在 `wan_offline_tools/`。

训练使用 uniform32 抽帧。缓存和元数据需要使用同一份帧列表；不要删减原始元数据后继续复用旧缓存。

## Foresight 配帧

`prepare_foresight_data.py` 接收训练/验证标注路径和观测缓存 ID，为相同场景资产上的文本生成任务建立留出片段清单，不限制数据集名称。已有 ScanQA 清单无需重建。参数和格式见 [foresight](../../../docs/foresight.md)。
