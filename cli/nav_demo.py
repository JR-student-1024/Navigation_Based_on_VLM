import argparse, json
from nav_vlm.core.pipeline import run
from nav_vlm.adapters.qwen_narrator import QwenNarrator
import os, pathlib

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--depth-ckpt", required=True)
    ap.add_argument("--sam2-ckpt", required=True)
    ap.add_argument("--sam2-cfg", required=True)
    ap.add_argument("--lang", default="ENG")
    ap.add_argument("--outdir", default=r"D:\Python\Python Project\nav_vlm\outputs")
    ap.add_argument("--gray", action="store_true", help="Save as a grayscale depth map (default pseudocolor)")
    args = ap.parse_args()

    # path：{filename}_depth.png
    os.makedirs(args.outdir, exist_ok=True)
    stem = pathlib.Path(args.img).stem
    out_depth = os.path.join(args.outdir, f"{stem}_depth.png")

    summary = run(
        image_path=args.img,
        ckpt_depth=args.depth_ckpt,
        ckpt_sam2=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
        lang=args.lang,
        save_depth_png = out_depth,
        color_depth = not args.gray
    )
    print(f"\nDepth saved to: {out_depth}")
    print("JSON:", json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    text = QwenNarrator().speak(args.img, summary, args.lang)
    print("\nVLM:", text)

