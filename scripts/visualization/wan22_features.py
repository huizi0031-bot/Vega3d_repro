#!/usr/bin/env python3
"""Visualize cached Wan2.2 features without loading model weights or using CUDA."""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-file', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--source-patch', type=int, nargs=2, default=[7, 7], metavar=('ROW', 'COL'))
    parser.add_argument('--similarity-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Fixed cosine color limits; default uses the joint observed range')
    parser.add_argument('--output', type=Path, default=Path('results/visualization/wan_features.png'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.chdir(REPO)
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if (args.output.exists() or args.output.with_suffix('.json').exists()) and not args.overwrite:
        raise FileExistsError('Choose another output or use --overwrite')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    payload = torch.load(args.feature_file, map_location='cpu', weights_only=True)
    features = payload['features']
    if features.ndim != 4 or not torch.isfinite(features).all():
        raise ValueError('Expected finite [frames, channels, height, width] features')
    count, channels, height, width = features.shape
    if any(index < 0 or index >= count for index in args.frames):
        raise ValueError('Frame index is outside the cached frame range')
    row, col = args.source_patch
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError('Source patch is outside the feature grid')
    selected = features[args.frames].float()
    flat = selected.permute(0, 2, 3, 1).reshape(-1, channels)
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False, niter=3)
    colors = (centered @ basis).reshape(2, height, width, 3).numpy()
    lo = np.percentile(colors, 2, axis=(0, 1, 2), keepdims=True)
    hi = np.percentile(colors, 98, axis=(0, 1, 2), keepdims=True)
    colors = np.clip((colors - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    normalized = F.normalize(selected, dim=1)
    similarities = torch.einsum('c,nchw->nhw', normalized[0, :, row, col], normalized).numpy()
    similarities = np.clip(similarities, -1, 1)
    limits = args.similarity_range or [float(similarities.min()), float(similarities.max())]
    if args.similarity_range and not (-1 <= limits[0] < limits[1] <= 1):
        raise ValueError('Similarity range must satisfy -1 <= MIN < MAX <= 1')
    if limits[0] == limits[1]:
        limits = [max(-1, limits[0] - .05), min(1, limits[1] + .05)]
    frame_files = payload['metadata']['frame_files']
    if len(frame_files) != count:
        raise ValueError('Frame metadata does not match the feature count')
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    images = []
    for view, index in enumerate(args.frames):
        path = Path(frame_files[index])
        path = path if path.is_absolute() else REPO/path
        with Image.open(path) as image:
            axes[view, 0].imshow(image.convert('RGB'))
        images.append(str(path))
        axes[view, 0].set_title(f'Observed frame {index}')
        axes[view, 1].imshow(colors[view], interpolation='nearest')
        axes[view, 1].set_title('Joint feature PCA')
        heatmap = axes[view, 2].imshow(similarities[view], vmin=limits[0], vmax=limits[1], cmap='viridis')
        axes[view, 2].set_title('Cosine similarity to source patch')
        for axis in axes[view]:
            axis.axis('off')
    for axis in [axes[0, 1], axes[0, 2]]:
        axis.scatter([col], [row], facecolors='none', edgecolors='red', s=90, linewidths=1.5)
    figure.colorbar(heatmap, ax=axes[:, 2], shrink=.75)
    figure.suptitle('Wan2.2 offline features')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    record = dict(feature_file=str(args.feature_file.resolve()), feature_shape=list(features.shape),
                  frames=args.frames, frame_files=images, source_patch=args.source_patch,
                  similarity_range=limits,
                  cuda_initialized=torch.cuda.is_initialized(),
                  interpretation='Qualitative feature visualization; does not establish geometric correspondence.')
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
