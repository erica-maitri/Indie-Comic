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
    parser.add_argument("--ref_img", type=str, required=True, help="Path to reference ground-truth image (for FID, DINOv2, CLIP)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt used to generate image (for CLIP Text-Image)")
    parser.add_argument("--gen_text", type=str, default="", help="Generated dialogue/text")
    parser.add_argument("--ref_text", type=str, default="", help="Reference dialogue/text")
    parser.add_argument("--gen_bbox", type=str, default="", help="Generated bounding box in format x1,y1,x2,y2 (for IoU)")
    parser.add_argument("--ref_bbox", type=str, default="", help="Reference bounding box in format x1,y1,x2,y2 (for IoU)")
    
    args = parser.parse_args()
    
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
        
    clip_img_score = evaluator.compute_clip_image_similarity(gen_img, ref_img)
    if clip_img_score is not None:
        metrics['CLIP Img2Img'] = clip_img_score
        print(f"  -> CLIP Img-Img:    {metrics['CLIP Img2Img']:.4f} (higher is better)")
        
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
        print("\n[5] Layout & Bounding Box Quality")
        try:
            gen_box = tuple(map(int, args.gen_bbox.split(',')))
            ref_box = tuple(map(int, args.ref_bbox.split(',')))
            iou_score = evaluator.compute_iou(gen_box, ref_box)
            metrics['IoU'] = iou_score
            print(f"  -> IoU Score:       {metrics['IoU']:.4f} (higher is better)")
        except Exception as e:
            print(f"  -> Error computing IoU: {e}")

    print("\n" + "="*50)
    print(" Final JSON Report")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    
    # Free memory
    evaluator.free_memory()

if __name__ == "__main__":
    main()
