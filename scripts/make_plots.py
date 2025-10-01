#!/usr/bin/env python
"""
Run plotting after training in the same process so the models are in-memory.
"""
from src.evaluate import run_experiment  # returns (models, test_loader)
from src.viz.attn_plots import (
    plot_attn_heatmaps_per_model,
    plot_overlaid_attn_3d,
    plot_crossdecode_attn_3d,
    plot_cross_attn_heatmaps,
    plot_decoder_self_per_head
)

def main():
    models, _ = run_experiment(train_epochs=8)  # adjust epochs if you want
    text = "secure message"
    keys = ["M1", "M2", "M3"]

    # Heatmaps (encoder & decoder)
    plot_attn_heatmaps_per_model(models, text, keys, which="encoder", out_dir="plots", save_prefix="enc_heat")
    plot_attn_heatmaps_per_model(models, text, keys, which="decoder", out_dir="plots", save_prefix="dec_heat")

    # 3D overlays
    plot_overlaid_attn_3d(models, text, keys, style="wire", out_dir="plots", save_stem="overlay_wire")
    plot_crossdecode_attn_3d(models, text, "M1", keys, out_dir="plots", save_stem="crossdecode")

    # Per-head examples
    for k in keys + ["M1_CLONE", "M1_SAMESEED"]:
        if k in models:
            plot_decoder_self_per_head(models, text, k, out_dir="plots", save_prefix="dec_heads")

    # Cross-attn per-head heatmaps
    plot_cross_attn_heatmaps(models, text, "M1", keys, out_dir="plots", save_prefix="cross_heat")

if __name__ == "__main__":
    main()
