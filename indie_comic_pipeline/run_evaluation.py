import os
import sys
import json
from PIL import Image

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.evaluation_suite import ModelEvaluator

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation Suite")
    parser.add_argument("--gen_img", type=str, required=True, help="Path to generated image")
    parser.add_argument("--ref_img", type=str, required=False, help="Path to reference ground-truth image (for FID, DINOv2, CLIP)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt used to generate image (for CLIP Text-Image)")
    parser.add_argument("--gen_text", type=str, default="", help="Generated dialogue/text")
    parser.add_argument("--ref_text", type=str, default="", help="Reference dialogue/text")
    parser.add_argument("--gen_bbox", type=str, default="", help="Generated bounding box in format x1,y1,x2,y2 (for IoU)")
    parser.add_argument("--ref_bbox", type=str, default="", help="Reference bounding box in format x1,y1,x2,y2 (for IoU)")
    parser.add_argument("--pred_boxes", type=str, default="", help="Multiple predicted bounding boxes separated by semicolon, e.g., 'x1,y1,x2,y2;x3,y3,x4,y4'")
    parser.add_argument("--gt_boxes", type=str, default="", help="Multiple ground-truth bounding boxes separated by semicolon, e.g., 'x1,y1,x2,y2;x3,y3,x4,y4'")
    parser.add_argument("--pred_mask", type=str, default="", help="Path to predicted binary mask image (for SAM 2.1)")
    parser.add_argument("--gt_mask", type=str, default="", help="Path to ground-truth binary mask image (for SAM 2.1)")
    
    args = parser.parse_args()
    
    if not args.ref_img:
        # Default to Panel 1 for Character/Style Consistency checking (Option 1)
        args.ref_img = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "panels", "panel_001_final.png")
        print(f"[*] No --ref_img provided. Using Panel 1 as consistency anchor: {args.ref_img}")

    evaluator = ModelEvaluator()
    print("="*50)
    print(" Model Evaluation Suite")
    print("="*50)
    
    metrics = {}
    
    # Load images
    try:
        gen_img = Image.open(args.gen_img).convert("RGB")
        ref_img = Image.open(args.ref_img).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Image Quality
    print("\n[1] Image Quality & Realism")
    metrics['Aesthetic Score'] = evaluator.compute_aesthetic_score(gen_img)
    print(f"  -> Aesthetic Score: {metrics['Aesthetic Score']:.4f}")
    
    fid_score = evaluator.compute_fid(gen_img, ref_img)
    if fid_score is not None:
        metrics['FID'] = fid_score
        print(f"  -> FID Score:       {metrics['FID']:.4f} (lower is better)")

    # Semantic & Structural Consistency
    print("\n[2] Semantic & Structural Consistency")
    dinov2_score = evaluator.compute_dinov2_similarity(gen_img, ref_img)
    if dinov2_score is not None:
        metrics['DINOv2 Similarity'] = dinov2_score
        print(f"  -> DINOv2:          {metrics['DINOv2 Similarity']:.4f} (higher is better)")

    dinov3_score = evaluator.compute_dinov3_similarity(gen_img, ref_img)
    if dinov3_score is not None:
        metrics['DINOv3 Similarity'] = dinov3_score
        print(f"  -> DINOv3:          {metrics['DINOv3 Similarity']:.4f} (higher is better)")

    siglip_score = evaluator.compute_siglip_similarity(gen_img, ref_img)
    if siglip_score is not None:
        metrics['SigLIP Similarity'] = siglip_score
        print(f"  -> SigLIP Similarity: {metrics['SigLIP Similarity']:.4f} (higher is better)")
        
    clip_img_score = evaluator.compute_clip_image_similarity(gen_img, ref_img)
    if clip_img_score is not None:
        metrics['CLIP Img2Img'] = clip_img_score
        print(f"  -> CLIP Img-Img:    {metrics['CLIP Img2Img']:.4f} (higher is better)")

    lpips_score = evaluator.compute_lpips(gen_img, ref_img)
    if lpips_score is not None:
        metrics['LPIPS'] = lpips_score
        print(f"  -> LPIPS Perceptual: {metrics['LPIPS']:.4f} (lower is better)")
        
    ssim_score = evaluator.compute_ssim(gen_img, ref_img)
    if ssim_score is not None:
        metrics['SSIM'] = ssim_score
        print(f"  -> SSIM:            {metrics['SSIM']:.4f} (higher is better)")

    psnr_score = evaluator.compute_psnr(gen_img, ref_img)
    if psnr_score is not None:
        metrics['PSNR'] = psnr_score
        print(f"  -> PSNR:            {metrics['PSNR']:.4f} (higher is better)")

    # Text-to-Image Alignment
    if args.prompt:
        print("\n[3] Text-to-Image Alignment")
        clip_text_score = evaluator.compute_clip_text_alignment(gen_img, args.prompt)
        if clip_text_score is not None:
            metrics['CLIP Text2Img'] = clip_text_score
            print(f"  -> CLIP Text-Img:   {metrics['CLIP Text2Img']:.4f} (higher is better)")

    # Text Generation Quality
    if args.gen_text and args.ref_text:
        print("\n[4] Text Generation Quality")
        bleu_score = evaluator.compute_bleu(args.gen_text, args.ref_text)
        if bleu_score is not None:
            metrics['BLEU Score'] = bleu_score
            print(f"  -> BLEU:            {metrics['BLEU Score']:.4f} (higher is better)")

    # Bounding Box Evaluation
    if args.gen_bbox and args.ref_bbox:
        print("\n[5] Single Bounding Box Layout Quality")
        try:
            vals_gen = list(map(int, args.gen_bbox.split(',')))
            vals_ref = list(map(int, args.ref_bbox.split(',')))
            if len(vals_gen) == 4 and len(vals_ref) == 4:
                gen_box = (vals_gen[0], vals_gen[1], vals_gen[2], vals_gen[3])
                ref_box = (vals_ref[0], vals_ref[1], vals_ref[2], vals_ref[3])
                iou_score = evaluator.compute_iou(gen_box, ref_box)
                metrics['IoU'] = iou_score
                print(f"  -> IoU Score:       {metrics['IoU']:.4f} (higher is better)")
            else:
                print("  -> Error: Bounding boxes must have exactly 4 values (x1,y1,x2,y2)")
        except Exception as e:
            print(f"  -> Error computing IoU: {e}")
 
    # Multiple Speech Bubble Detection Metrics (Grounding DINO)
    if args.pred_boxes and args.gt_boxes:
        print("\n[6] Speech Bubble Detection Metrics (Grounding DINO 1.5)")
        try:
            p_boxes = []
            for b in args.pred_boxes.split(';'):
                if b.strip():
                    vals = list(map(int, b.split(',')))
                    if len(vals) == 4:
                        p_boxes.append((vals[0], vals[1], vals[2], vals[3]))
            g_boxes = []
            for b in args.gt_boxes.split(';'):
                if b.strip():
                    vals = list(map(int, b.split(',')))
                    if len(vals) == 4:
                        g_boxes.append((vals[0], vals[1], vals[2], vals[3]))
            
            det_metrics = evaluator.compute_detection_metrics(p_boxes, g_boxes)
            for k, v in det_metrics.items():
                metrics[f'Bubble Detection {k}'] = v
                print(f"  -> Bubble Detection {k}: {v:.4f}")
        except Exception as e:
            print(f"  -> Error computing bubble detection metrics: {e}")

    # Character Segmentation Metrics (SAM 2.1)
    if args.pred_mask and args.gt_mask:
        print("\n[7] Character Segmentation Metrics (SAM 2.1)")
        try:
            import numpy as np
            p_mask = np.array(Image.open(args.pred_mask).convert("L")) > 127
            g_mask = np.array(Image.open(args.gt_mask).convert("L")) > 127
            
            seg_metrics = evaluator.compute_segmentation_metrics(p_mask, g_mask)
            for k, v in seg_metrics.items():
                metrics[f'Character Seg {k}'] = v
                print(f"  -> Character Segmentation {k}: {v:.4f}")
        except Exception as e:
            print(f"  -> Error computing character segmentation metrics: {e}")

    print("\n" + "="*50)
    print(" Final JSON Report")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    
    # Free memory
    evaluator.free_memory()

if __name__ == "__main__":
    main()
