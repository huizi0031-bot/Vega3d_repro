# Wan2.2 特征可视化

`wan22_features.py` 读取当前 `wan22_ti2v.pt`，不加载 Qwen 或 Wan 权重，不使用 GPU。每行显示一帧 RGB、两帧联合计算的特征 PCA，以及所选 source patch 对各帧 patch 的余弦相似度。

```bash
python scripts/visualization/wan22_features.py \
  --feature-file data/scannet/wan22_ti2v_offline_uniform32_1280x704_t300_b20_s42_v1/scene0281_00/wan22_ti2v.pt \
  --frames 0 1 --source-patch 7 7 \
  --output results/visualization/wan_features.png
```

`--frames` 使用缓存中的帧索引，不是原始扫描帧编号。JSON 会记录真实图像路径、特征形状和选取的 patch。红圈是第一帧所选 patch；PCA 颜色由两帧共享的投影得到。RGB 仅作原图参考，未覆盖几何重投影。

该工具用于查看特征分布及相似度，不能直接把高相似度当成正确几何匹配。原版的框投影、多视角对应和三视角视频工具依赖旧 Wan2.1 路径，尚未移植；原实现可从 Git 历史恢复，当前不将其列为已适配功能。

两行热图共用色标，默认按两帧的实际数值范围显示，范围同时写入 JSON。跨场景比较时可指定 `--similarity-range -1 1` 使用固定范围；注意读取色标数值，不要只比较颜色。
