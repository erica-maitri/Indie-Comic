import os
import time
import json


class ModelComparator:
    """Utility to A/B test different diffusion models and LoRAs to quantify performance"""
    
    def __init__(self, output_dir="outputs/comparison"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        from core.evaluation_suite import ModelEvaluator
        self.metrics = ModelEvaluator()
        
    def compare_models(self, prompt: str, models: dict, reference_image=None):
        """
        Runs the prompt through a dictionary of initialized pipelines/functions.
        models = {'SDXL': sdxl_generate_func, 'SD1.5': sd15_generate_func}
        """
        results = {}
        
        for model_name, model_fn in models.items():
            print(f"[*] Testing {model_name}...")
            start_time = time.time()
            
            try:
                # Generate image
                output_image = model_fn(prompt)
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Save image
                safe_name = model_name.replace(" ", "_").lower()
                img_path = os.path.join(self.output_dir, f"{safe_name}_output.png")
                output_image.save(img_path)
                
                # Calculate metrics
                fid_score = None
                if reference_image:
                    fid_score = self.metrics.compute_fid(output_image, reference_image)
                    
                # Store stats
                results[model_name] = {
                    'image_path': img_path,
                    'time_seconds': round(generation_time, 2),
                    'fid': fid_score if fid_score else "N/A",
                    'file_size_kb': round(os.path.getsize(img_path) / 1024, 2)
                }
                
                print(f"[✓] {model_name} finished in {generation_time:.2f}s")
                
            except Exception as e:
                print(f"[!] {model_name} failed: {e}")
                results[model_name] = {'error': str(e)}
                
        self.generate_report(prompt, results)
        return results
        
    def generate_report(self, prompt: str, results: dict):
        """Generates an HTML and JSON report of the model comparison"""
        # Save JSON
        json_path = os.path.join(self.output_dir, "report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'prompt': prompt, 'results': results}, f, indent=4)
            
        # Save HTML
        html_path = os.path.join(self.output_dir, "report.html")
        html_content = f"""
        <html>
        <head><title>Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #121212; color: #fff; padding: 20px; }}
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .card {{ background: #1e1e1e; padding: 15px; border-radius: 8px; max-width: 400px; border: 1px solid #333; }}
            img {{ max-width: 100%; border-radius: 4px; }}
            .metric {{ color: #bb86fc; font-weight: bold; }}
        </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
            <p><strong>Prompt:</strong> "{prompt}"</p>
            <div class="grid">
        """
        
        for name, data in results.items():
            html_content += f'<div class="card"><h3>{name}</h3>'
            if 'error' in data:
                html_content += f'<p style="color: red;">Error: {data["error"]}</p>'
            else:
                img_src = os.path.basename(data["image_path"])
                html_content += f'<img src="{img_src}" alt="{name} output">'
                html_content += f'<p>Time: <span class="metric">{data["time_seconds"]}s</span></p>'
                html_content += f'<p>FID Score: <span class="metric">{data["fid"]}</span></p>'
                html_content += f'<p>Size: <span class="metric">{data["file_size_kb"]} KB</span></p>'
            html_content += '</div>'
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"[*] Comparison report generated: {html_path}")
