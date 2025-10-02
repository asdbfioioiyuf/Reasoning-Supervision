"""
Training script for video action recognition
Supports multi-scale temporal reasoning and spatial attention supervision
"""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import VideoActionRecognitionModel
from atten import spatial_kl_loss, build_gt_mask_from_batch_meta, spatial_attn_to_probs


def compute_js_divergence_loss(p_immediate, p_contextual, p_narrative, temperature=1.0):
    """
    Compute Jensen-Shannon divergence loss for multi-view agreement.

    Args:
        p_immediate: [B, num_classes] Probability distribution from immediate view
        p_contextual: [B, num_classes] Probability distribution from contextual view
        p_narrative: [B, num_classes] Probability distribution from narrative view
        temperature: Temperature for softening the consensus distribution

    Returns:
        JS divergence loss (scalar)
    """
    p_bar = (p_immediate + p_contextual + p_narrative) / 3.0

    if temperature != 1.0:
        p_bar = torch.pow(p_bar, 1.0 / temperature)
        p_bar = p_bar / (p_bar.sum(dim=-1, keepdim=True) + 1e-12)

    kl_immediate = torch.nn.functional.kl_div(
        torch.log(p_bar + 1e-12), p_immediate, reduction='batchmean'
    )
    kl_contextual = torch.nn.functional.kl_div(
        torch.log(p_bar + 1e-12), p_contextual, reduction='batchmean'
    )
    kl_narrative = torch.nn.functional.kl_div(
        torch.log(p_bar + 1e-12), p_narrative, reduction='batchmean'
    )

    js_loss = (kl_immediate + kl_contextual + kl_narrative) / 3.0
    return js_loss


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
    lambda_spatial=0.2,
    lambda_js=0.3,
    spatial_supervision=True,
    attn_every=10,
    js_temperature=0.7,
    accumulation_steps=1,
    grid_h=4,
    grid_w=4
):
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs
        lambda_spatial: Weight for spatial supervision loss
        lambda_js: Weight for JS divergence loss
        spatial_supervision: Whether to use spatial supervision
        attn_every: Compute attention every N steps
        js_temperature: Temperature for JS divergence
        accumulation_steps: Gradient accumulation steps
        grid_h, grid_w: Spatial grid dimensions

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    seen = 0

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    global_step = 0

    N = grid_h * grid_w

    last_spatial_loss = 0.0
    last_js_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{total_epochs}]")
    accumulation_counter = 0

    for clips, labels, metas in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B, T = clips.shape[:2]

        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True)

        want_attn = (attn_every is not None) and (global_step % attn_every == 0)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
            out = model(clips, return_attn=want_attn)

            # Multi-view classification loss
            immediate_logits = out["immediate_logits"]
            contextual_logits = out["contextual_logits"]
            narrative_logits = out["narrative_logits"]

            cls_immediate = criterion(immediate_logits, labels)
            cls_contextual = criterion(contextual_logits, labels)
            cls_narrative = criterion(narrative_logits, labels)

            per_view_cls_loss = cls_immediate + cls_contextual + cls_narrative

            # JS divergence loss
            immediate_probs = torch.softmax(immediate_logits, dim=-1)
            contextual_probs = torch.softmax(contextual_logits, dim=-1)
            narrative_probs = torch.softmax(narrative_logits, dim=-1)

            js_loss = compute_js_divergence_loss(
                immediate_probs, contextual_probs, narrative_probs,
                temperature=js_temperature
            )
            last_js_loss = js_loss.item()

            # Spatial supervision loss
            spatial_loss = clips.new_tensor(0.0)
            if want_attn and spatial_supervision:
                attn_sp = out.get("attn_spatial", None)
                if attn_sp is not None:
                    try:
                        # Build ground truth masks
                        M, Fmask, has_gt = build_gt_mask_from_batch_meta(metas, T, grid_h, grid_w)
                        M = M.to(device)
                        Fmask = Fmask.to(device)
                        has_gt = has_gt.to(device)

                        # Convert attention to probabilities
                        p = spatial_attn_to_probs(attn_sp, B, T, N, reduce="mean")

                        # Compute spatial KL loss
                        spatial_loss = spatial_kl_loss(
                            p=p,
                            M=M,
                            sample_mask=has_gt,
                            frame_mask=Fmask
                        )
                        last_spatial_loss = spatial_loss.item()
                    except Exception:
                        spatial_loss = clips.new_tensor(0.0)
                        last_spatial_loss = 0.0

            # Total loss
            total_loss = (
                per_view_cls_loss +
                lambda_js * js_loss +
                lambda_spatial * spatial_loss
            )

            # Scale by accumulation steps
            total_loss = total_loss / accumulation_steps

        # Backward pass
        scaler.scale(total_loss).backward()

        accumulation_counter += 1

        # Update weights after accumulation
        if accumulation_counter >= accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            accumulation_counter = 0
            global_step += 1

        running_loss += (total_loss.item() * accumulation_steps) * labels.size(0)
        seen += labels.size(0)

        # Update progress bar
        postfix_dict = {"total": f"{running_loss/seen:.4f}"}
        postfix_dict["js"] = f"{last_js_loss:.3f}"
        if spatial_supervision and last_spatial_loss > 0:
            postfix_dict["spatial"] = f"{last_spatial_loss:.3f}"

        pbar.set_postfix(postfix_dict)

    epoch_loss = running_loss / max(1, seen)
    return epoch_loss


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs=15,
    out_dir="checkpoints",
    lambda_spatial=0.2,
    lambda_js=0.3,
    spatial_supervision=True,
    attn_every=10,
    save_name="model.pth",
    js_temperature=0.7,
    accumulation_steps=1,
    grid_h=4,
    grid_w=4
):
    """
    Main training loop.

    Args:
        model: Model to train
        train_loader: DataLoader for training
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epochs: Number of epochs
        out_dir: Output directory for checkpoints
        lambda_spatial: Spatial loss weight
        lambda_js: JS divergence loss weight
        spatial_supervision: Whether to use spatial supervision
        attn_every: Compute attention every N steps
        save_name: Checkpoint filename
        js_temperature: Temperature for JS divergence
        accumulation_steps: Gradient accumulation steps
        grid_h, grid_w: Spatial grid dimensions

    Returns:
        List of training losses
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            lambda_spatial=lambda_spatial,
            lambda_js=lambda_js,
            spatial_supervision=spatial_supervision,
            attn_every=attn_every,
            js_temperature=js_temperature,
            accumulation_steps=accumulation_steps,
            grid_h=grid_h,
            grid_w=grid_w
        )

        loss_history.append(epoch_loss)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_dir / save_name)
            print(f"New best model saved: {best_loss:.4f}")

        print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")

    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    return loss_history


def main():
    parser = argparse.ArgumentParser(description='Train video action recognition model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--anno_path', type=str, default=None, help='Path to annotation JSON file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=101, help='Number of classes')
    parser.add_argument('--T', type=int, default=64, help='Number of temporal frames')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--lambda_spatial', type=float, default=0.15, help='Spatial loss weight')
    parser.add_argument('--lambda_js', type=float, default=0.25, help='JS loss weight')
    parser.add_argument('--no_spatial', action='store_true', help='Disable spatial supervision')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model
    model = VideoActionRecognitionModel(
        feature_dim=2048,
        spatial_size=4,
        embed_dim=128,
        spatial_layers=4,
        spatial_heads=8,
        temporal_layers=3,
        temporal_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_T=args.T,
        num_classes=args.num_classes
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # TODO: Replace with your own DataLoader
    # train_loader = create_dataloader(args.data_dir, args.batch_size, args.T)
    print("Please implement your own DataLoader in this script")
    print("See README.md for dataset format requirements")
    return

    # Start training
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        out_dir=args.out_dir,
        lambda_spatial=args.lambda_spatial,
        lambda_js=args.lambda_js,
        spatial_supervision=not args.no_spatial,
        attn_every=10,
        save_name="model.pth",
        js_temperature=0.7,
        accumulation_steps=8,
        grid_h=4,
        grid_w=4
    )

  


if __name__ == "__main__":
    main()
