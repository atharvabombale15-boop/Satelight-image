import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import LEVIRDataset
from config import MODEL_ARCH
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate optimal threshold on validation set.")
    parser.add_argument("--data", default="data", help="Dataset root path")
    parser.add_argument("--model", default="models/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--image-size", type=int, default=256, help="Inference image size")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--metric", choices=["f1", "iou"], default="f1", help="Metric to optimize")
    parser.add_argument("--min-threshold", type=float, default=0.05, help="Minimum threshold")
    parser.add_argument("--max-threshold", type=float, default=0.95, help="Maximum threshold")
    parser.add_argument("--steps", type=int, default=19, help="Number of thresholds to test")
    return parser.parse_args()


def compute_scores(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    val_dataset = LEVIRDataset(args.data, split="val", image_size=args.image_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(MODEL_ARCH).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    thresholds = torch.linspace(args.min_threshold, args.max_threshold, args.steps, device=device)
    tp = torch.zeros_like(thresholds)
    fp = torch.zeros_like(thresholds)
    fn = torch.zeros_like(thresholds)

    with torch.no_grad():
        for t1, t2, mask in val_loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)
            preds = model(t1, t2)

            preds_flat = preds.view(preds.size(0), -1)
            mask_flat = mask.view(mask.size(0), -1)

            for i, thr in enumerate(thresholds):
                pred_bin = (preds_flat > thr).float()
                tp[i] += (pred_bin * mask_flat).sum()
                fp[i] += (pred_bin * (1 - mask_flat)).sum()
                fn[i] += ((1 - pred_bin) * mask_flat).sum()

    precision, recall, f1, iou = compute_scores(tp, fp, fn)
    metric = f1 if args.metric == "f1" else iou
    best_idx = int(torch.argmax(metric).item())
    best_thr = float(thresholds[best_idx].item())

    print("Threshold calibration results:")
    for i, thr in enumerate(thresholds):
        print(
            f"thr={thr:.3f} "
            f"precision={precision[i]:.4f} "
            f"recall={recall[i]:.4f} "
            f"f1={f1[i]:.4f} "
            f"iou={iou[i]:.4f}"
        )

    print("\nBest threshold:")
    print(f"metric={args.metric} threshold={best_thr:.3f} score={metric[best_idx]:.4f}")


if __name__ == "__main__":
    main()
