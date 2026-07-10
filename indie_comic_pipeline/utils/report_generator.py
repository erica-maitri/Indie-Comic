"""
REPORT GENERATOR — Benchmark HTML Dashboard Renderer
=====================================================
Generates an interactive, self-contained HTML report from benchmark results.
Includes inline SVG performance charts and a side-by-side comparison slider.
"""

import os
import json
import logging
from typing import Dict, Any, List

log = logging.getLogger("pipeline.benchmark.report")


class ReportGenerator:
    """
    Renders benchmark results into a premium, responsive dark-mode HTML page.
    Includes custom SVG visualizers to prevent external JS/CSS dependencies.
    """

    @staticmethod
    def generate_html_report(results: List[Dict[str, Any]], 
                             recommendation: Dict[str, Any], 
                             output_path: str) -> bool:
        """
        Generates the standalone HTML report.
        
        Args:
            results: List of run results dictionaries from BenchmarkSuite.
            recommendation: Recommended optimal configuration dictionary.
            output_path: Target path to write the HTML file.
            
        Returns:
            True if successfully generated, False otherwise.
        """
        try:
            # 1. Generate SVG Chart for Generation Time vs. Resolution
            time_chart_svg = ReportGenerator._render_time_chart(results)

            # 2. Generate SVG Chart for Peak VRAM vs. Resolution
            vram_chart_svg = ReportGenerator._render_vram_chart(results)

            # 3. Build comparison image selections
            img_options = "".join([
                f'<option value="{run["image_name"]}">{run["resolution"]}x{run["resolution"]} - {run["steps"]} steps - LoRA {run["lora_scale"]}</option>'
                for run in results if not run.get("error")
            ])

            # Use baseline or default if no runs succeeded
            baseline_img = "baseline_image.png"
            first_run_img = results[0]["image_name"] if results else "baseline_image.png"

            # 4. Generate HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indie Comic Pipeline - Benchmark Report</title>
    <style>
        :root {{
            --bg-dark: #0a0b10;
            --bg-card: #12131a;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --success: #10b981;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border: #1f2937;
        }}

        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, sans-serif;
            margin: 0;
            padding: 2rem;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }}

        h1 {{
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}

        .card h2 {{
            margin-top: 0;
            font-size: 1.25rem;
            color: #a5b4fc;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}

        /* Recommendation Banner */
        .recommendation-banner {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(192, 132, 252, 0.1) 100%);
            border: 1px dashed var(--primary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .rec-info h3 {{
            margin: 0;
            color: #818cf8;
            font-size: 1.4rem;
        }}

        .rec-info p {{
            margin: 0.5rem 0 0 0;
            color: var(--text-muted);
        }}

        .rec-stats {{
            display: flex;
            gap: 2rem;
        }}

        .stat-box {{
            text-align: center;
        }}

        .stat-val {{
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--success);
        }}

        .stat-lbl {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        /* Visual Comparison Slider */
        .comparison-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 2rem;
        }}

        .slider-selectors {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            width: 100%;
            justify-content: center;
        }}

        .slider-selectors select {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.9rem;
        }}

        .image-compare-wrapper {{
            position: relative;
            width: 600px;
            height: 600px;
            border: 2px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            user-select: none;
        }}

        .compare-img {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .compare-img-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);
        }}

        .slider-handle {{
            position: absolute;
            top: 0;
            bottom: 0;
            left: 50%;
            width: 4px;
            background-color: var(--primary);
            cursor: ew-resize;
            z-index: 10;
        }}

        .slider-handle::after {{
            content: "↔";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 32px;
            height: 32px;
            background-color: var(--primary);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }}

        /* Table styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        th, td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}

        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        .badge-err {{
            background-color: #ef4444;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
        }}

        /* SVG charts */
        .chart-container {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Indie Comic Pipeline</h1>
            <div class="subtitle">Hyperparameter Tuning & Benchmarking Dashboard</div>
        </header>

        <!-- Recommendation Panel -->
        <div class="recommendation-banner">
            <div class="rec-info">
                <h3>Recommended Optimal Setting</h3>
                <p>The recommendation engine selects the setup with the best quality-to-speed ratio.</p>
            </div>
            <div class="rec-stats">
                <div class="stat-box">
                    <div class="stat-val">{recommendation.get("resolution", "768")}x{recommendation.get("resolution", "768")}</div>
                    <div class="stat-lbl">Resolution</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{recommendation.get("steps", "25")}</div>
                    <div class="stat-lbl">Inference Steps</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{recommendation.get("lora_scale", "0.8")}</div>
                    <div class="stat-lbl">LoRA Scale</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #6366f1;">{recommendation.get("generation_time_s", "0.0")}s</div>
                    <div class="stat-lbl">Latency</div>
                </div>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="grid-2">
            <div class="card">
                <h2>Generation Time vs Configuration</h2>
                <div class="chart-container">
                    {time_chart_svg}
                </div>
            </div>
            <div class="card">
                <h2>VRAM Peak Memory Consumption</h2>
                <div class="chart-container">
                    {vram_chart_svg}
                </div>
            </div>
        </div>

        <!-- Slider Visual Comparison -->
        <div class="card comparison-container">
            <h2>Interactive Image Quality Comparison</h2>
            <div class="slider-selectors">
                <div>
                    <label>Left Image (Baseline): </label>
                    <select id="leftSelect" disabled>
                        <option>Baseline Image (Highest quality)</option>
                    </select>
                </div>
                <div>
                    <label>Right Image (Compare): </label>
                    <select id="rightSelect" onchange="updateRightImage(this.value)">
                        {img_options}
                    </select>
                </div>
            </div>
            <div class="image-compare-wrapper" id="compareWrapper">
                <!-- Left (Baseline) Image -->
                <img src="{baseline_img}" class="compare-img" alt="Baseline image">
                <!-- Right (Compare) Image -->
                <img src="{first_run_img}" class="compare-img-overlay" id="compareOverlay" alt="Compare image">
                <!-- Slider divider line -->
                <div class="slider-handle" id="sliderHandle"></div>
            </div>
        </div>

        <!-- Sweep Results Table -->
        <div class="card">
            <h2>All Grid Sweep Results ({len(results)} runs)</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Resolution</th>
                        <th>Steps</th>
                        <th>LoRA Scale</th>
                        <th>Time (s)</th>
                        <th>Peak VRAM (MB)</th>
                        <th>SSIM</th>
                        <th>Edge Sim</th>
                        <th>Color Sim</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """

            # Add rows to table
            for run in results:
                err_badge = '<span class="badge-err">Error</span>' if run.get("error") else '<span style="color: var(--success);">Success</span>'
                html_content += f"""
                    <tr>
                        <td>{run["run_id"]}</td>
                        <td>{run["resolution"]}x{run["resolution"]}</td>
                        <td>{run["steps"]}</td>
                        <td>{run["lora_scale"]}</td>
                        <td>{run["generation_time_s"]}s</td>
                        <td>{run["vram_peak_mb"]} MB</td>
                        <td>{run["ssim_similarity"]}</td>
                        <td>{run["edge_similarity"]}</td>
                        <td>{run["color_similarity"]}</td>
                        <td>{err_badge}</td>
                    </tr>
                """

            # Close HTML template
            html_content += """
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Custom Image Slider logic
        const wrapper = document.getElementById('compareWrapper');
        const overlay = document.getElementById('compareOverlay');
        const handle = document.getElementById('sliderHandle');
        let isDragging = false;

        function setSliderPosition(x) {
            const rect = wrapper.getBoundingClientRect();
            let pos = (x - rect.left) / rect.width;
            pos = Math.max(0, Math.min(1, pos)); // limit range between 0 and 1
            
            overlay.style.clipPath = `polygon(0 0, ${pos * 100}% 0, ${pos * 100}% 100%, 0 100%)`;
            handle.style.left = `${pos * 100}%`;
        }

        wrapper.addEventListener('mousedown', (e) => {
            isDragging = true;
            setSliderPosition(e.clientX);
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            setSliderPosition(e.clientX);
        });

        // Touch support
        wrapper.addEventListener('touchstart', (e) => {
            isDragging = true;
            setSliderPosition(e.touches[0].clientX);
        });

        window.addEventListener('touchend', () => {
            isDragging = false;
        });

        window.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            setSliderPosition(e.touches[0].clientX);
        });

        // Dropdown selection update
        function updateRightImage(imageName) {
            overlay.src = imageName;
        }
    </script>
</body>
</html>
"""
            # Write to output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            log.info(f"Interactive HTML report generated successfully at {output_path}")
            return True

        except Exception as e:
            log.error(f"Failed to generate HTML report: {e}")
            return False

    @staticmethod
    def _render_time_chart(results: List[Dict[str, Any]]) -> str:
        """Render inline SVG bar chart for generation latency."""
        valid_runs = [r for r in results if not r.get("error")]
        if not valid_runs:
            return '<svg width="400" height="200"><text x="100" y="100" fill="white">No data available</text></svg>'

        # Extract values
        times = [r["generation_time_s"] for r in valid_runs]
        max_time = max(times) if times else 1.0
        
        svg_w = 500
        svg_h = 250
        padding = 40
        chart_w = svg_w - (padding * 2)
        chart_h = svg_h - (padding * 2)
        
        bar_w = max(5, int(chart_w / len(valid_runs)) - 6)
        
        svg = f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color: #12131a; border-radius: 8px;">'
        
        # Gridlines
        for i in range(5):
            y = padding + chart_h - (i * chart_h / 4)
            val = i * max_time / 4
            svg += f'<line x1="{padding}" y1="{y}" x2="{svg_w - padding}" y2="{y}" stroke="#1f2937" stroke-width="1"/>'
            svg += f'<text x="{padding - 10}" y="{y + 4}" fill="#9ca3af" font-size="10" text-anchor="end">{val:.1f}s</text>'
            
        # Draw bars
        for idx, run in enumerate(valid_runs):
            x = padding + idx * (bar_w + 6)
            h = (run["generation_time_s"] / max_time) * chart_h
            y = padding + chart_h - h
            
            # Use color based on steps configuration
            color = "#818cf8" if run["steps"] <= 20 else "#6366f1"
            if run["resolution"] > 512:
                color = "#a5b4fc"
                
            svg += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="3"/>'
            # Hover tooltip label
            svg += f'<title>Run {run["run_id"]}: {run["generation_time_s"]}s ({run["resolution"]}px, {run["steps"]} steps)</title>'
            
        svg += f'<text x="{svg_w//2}" y="{svg_h - 10}" fill="#9ca3af" font-size="11" text-anchor="middle">Sweeps Configurations Grid</text>'
        svg += '</svg>'
        return svg

    @staticmethod
    def _render_vram_chart(results: List[Dict[str, Any]]) -> str:
        """Render inline SVG bar chart for peak VRAM memory usage."""
        valid_runs = [r for r in results if not r.get("error")]
        # Extract VRAM peaks
        vrams = [r["vram_peak_mb"] for r in valid_runs]
        max_vram = max(vrams) if vrams and max(vrams) > 0 else 100.0
        
        svg_w = 500
        svg_h = 250
        padding = 45
        chart_w = svg_w - (padding * 2)
        chart_h = svg_h - (padding * 2)
        
        bar_w = max(5, int(chart_w / len(valid_runs)) - 6) if valid_runs else 20
        
        svg = f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color: #12131a; border-radius: 8px;">'
        
        # Gridlines
        for i in range(5):
            y = padding + chart_h - (i * chart_h / 4)
            val = i * max_vram / 4
            svg += f'<line x1="{padding}" y1="{y}" x2="{svg_w - padding}" y2="{y}" stroke="#1f2937" stroke-width="1"/>'
            svg += f'<text x="{padding - 10}" y="{y + 4}" fill="#9ca3af" font-size="10" text-anchor="end">{int(val)}MB</text>'
            
        # Draw bars
        for idx, run in enumerate(valid_runs):
            x = padding + idx * (bar_w + 6)
            h = (run["vram_peak_mb"] / max_vram) * chart_h if max_vram > 0 else 0
            y = padding + chart_h - h
            
            # Colors based on resolution (higher res = more VRAM)
            color = "#10b981" if run["resolution"] <= 512 else "#34d399"
            if run["vram_peak_mb"] == 0:
                color = "#374151" # Default grey if mock/CPU run
                h = 10
                y = padding + chart_h - h
                
            svg += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="3"/>'
            svg += f'<title>Run {run["run_id"]}: {run["vram_peak_mb"]} MB</title>'
            
        svg += f'<text x="{svg_w//2}" y="{svg_h - 10}" fill="#9ca3af" font-size="11" text-anchor="middle">Sweeps Configurations Grid</text>'
        svg += '</svg>'
        return svg
