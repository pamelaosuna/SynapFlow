import os

import numpy as np

from scipy.optimize import linear_sum_assignment
from skimage.exposure import rescale_intensity
from PIL import Image, ImageOps

import torch
from torchvision.ops.boxes import box_area
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


cmap = matplotlib.cm.get_cmap('tab20')
COLORS = [(int(cmap(i)[0]*255), int(cmap(i)[1]*255), int(cmap(i)[2]*255)) for i in range(20)]


# modified from torchvision to also return the union
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Code from https://github.com/timmeinhardt/trackformer
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Code from https://github.com/timmeinhardt/trackformer
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def hungarian_matching(cost_matrix: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return row_indices, col_indices

def draw_bbox_on_img_mpl(
        fig: plt.Figure, ax: plt.Axes, bbox: "list[float]", id: int, thickness: int = 2
        ) -> "tuple[plt.Figure, plt.Axes]":
    xmin, ymin, xmax, ymax = bbox
    
    color = COLORS[int(id)%20]
    # convert color to 0-1 range for matplotlib

    color = np.array(color) / 255.0
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
        linewidth=thickness, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    # ax.text(xmin + 2, ymin - 2, str(id), color=color, fontsize=8)
    
    return fig, ax

def draw_boxes_and_save(
        img: np.ndarray, bboxes: "list[list[float]]", ids: "list[int]", out_fp: str, thickness: int = 1
        ) -> np.ndarray:
    fig, ax = plt.subplots(1)
    img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    ax.imshow(img)
    plt.axis('off')

    for i, bbox in enumerate(bboxes):
        fig, ax = draw_bbox_on_img_mpl(fig, ax, bbox, ids[i], thickness)
        

    plt.tight_layout()
    plt.savefig(out_fp, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return img

def extract_embeddings(
        img_dir: str,
        im: dict,
        patch_size: "tuple[int, int]",
        boxes_orig: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
        ) -> torch.Tensor:
    """
    Extract bounding box region and convert into patch of size patch_size
    without distorting the aspect ratio (using padding). 
    Then pass through the model to get the embeddings.
    """
    img = Image.open(os.path.join(img_dir, im['file_name'])).convert('RGB')
    
    canvas = [Image.new('RGB', (patch_size, patch_size), (0, 0, 0)) for _ in boxes_orig]
    img_patch = [ImageOps.contain(img.crop(box), (patch_size, patch_size)) for box in boxes_orig]

    for c, p in zip(canvas, img_patch):
        c.paste(p, (0, 0))
    patches = torch.stack([transforms.ToTensor()(c) for c in canvas])

    with torch.no_grad():
        embeds = model.forward_once(patches.to(device))

    return embeds