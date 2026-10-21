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

foresight 训练直接读取原版处理后的标注，由加载层选择视频监督片段，不需要额外转换问答格式。Wan2.2 prompt 导出使用本目录的 `export_wan22_prompt_embedding.py`；离线特征读取和验证场景补齐入口在 `wan_offline_tools/`。

训练使用 uniform32 抽帧。缓存和元数据需要使用同一份帧列表；不要删减原始元数据后继续复用旧缓存。

## 可选配帧清单导出

`prepare_foresight_data.py` 是已有的离线筛选与清单导出工具，可用于固定子集或检查历史实验。完整训练无需运行该工具；默认入口保留全部标注，并在任何训练场景缺资源或无法配帧时明确报错。该工具会筛选符合条件的场景，不应把它的输出误认为未筛选的全量数据。
