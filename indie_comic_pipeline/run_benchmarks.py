#!/usr/bin/env python
"""
Indie Comic Pipeline - Benchmark & Tuning Launcher CLI
=====================================================
Usage:
  python run_benchmarks.py --mock
  python run_benchmarks.py --prompt "A cool robot" --apply
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline.benchmark.cli")

# Add current folder and core directories to system path for clean imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Indie Comic Pipeline - Benchmark Grid Sweeps & Tuning Harness")
    parser.add_argument("--prompt", type=str, default="A dark knight standing in the rain",
                        help="Prompt to test during parameter sweeps")
    parser.add_argument("--mock", action="store_true",
                        help="Run in CPU Mock mode (fast dry-run, no GPU or models needed)")
    parser.add_argument("--apply", action="store_true",
                        help="Automatically apply the recommended settings back to settings.yaml")
    parser.add_argument("--output", type=str, default=None,
                        help="Override default output directory for reports and images")
    args = parser.parse_args()

    try:
        from core.benchmark_suite import BenchmarkSuite
        from utils.report_generator import ReportGenerator
    except ImportError as e:
        log.error(f"Failed to import pipeline core modules: {e}")
        log.error("Please ensure you are running this script from the 'indie_comic_pipeline' root directory.")
        sys.exit(1)

    # Initialize benchmark suite
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    suite = BenchmarkSuite(settings_path=settings_path)

    # Override output directory if specified
    if args.output:
        suite.output_dir = args.output
        Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info("Starting Benchmark Sweep execution...")
    log.info(f"Target Prompt: '{args.prompt}'")
    log.info(f"Report Directory: {suite.output_dir}")

    # Run sweeps
    results = suite.run_sweeps(prompt=args.prompt, mock=args.mock)

    if not results:
        log.error("Benchmark sweep generated no results. Check logs for errors.")
        sys.exit(1)

    # Filter out errored runs
    successful_runs = [r for r in results if not r.get("error")]
    if not successful_runs:
        log.error("All benchmark sweep runs failed. No successful configurations found.")
        sys.exit(1)

    # Find recommended configuration
    recommendation = suite.get_recommendation(successful_runs)

    # Generate HTML report
    report_path = os.path.join(suite.output_dir, "benchmark_report.html")
    report_ok = ReportGenerator.generate_html_report(
        results=results,
        recommendation=recommendation,
        output_path=report_path
    )

    # Print summary leaderboard
    print("\n" + "=" * 80)
    print(f"🏆 BENCHMARK RESULTS LEADERBOARD ({'MOCK MODE' if args.mock else 'GPU MODE'})")
    print("=" * 80)
    print(f"{'Run ID':<8} | {'Resolution':<12} | {'Steps':<8} | {'LoRA':<6} | {'Latency':<9} | {'Peak VRAM':<12} | {'SSIM':<8}")
    print("-" * 80)
    for run in results:
        vram_str = f"{run['vram_peak_mb']:.1f} MB" if run['vram_peak_mb'] > 0 else "N/A (CPU)"
        status_suffix = "" if not run.get("error") else " ❌ (Err)"
        print(f"{run['run_id']:<8} | {f'{run['resolution']}x{run['resolution']}':<12} | {run['steps']:<8} | {run['lora_scale']:<6} | {run['generation_time_s']:<8}s | {vram_str:<12} | {run['ssim_similarity']:<8}{status_suffix}")
    print("=" * 80)

    # Print recommendation info
    if recommendation:
        print("\n💡 OPTIMIZATION RECOMMENDATION:")
        print(f"  • Optimal Configuration: {recommendation['resolution']}x{recommendation['resolution']} resolution, {recommendation['steps']} steps, LoRA scale {recommendation['lora_scale']}")
        print(f"  • Expected Generation Latency: {recommendation['generation_time_s']:.3f} seconds")
        if recommendation['vram_peak_mb'] > 0:
            print(f"  • Expected Peak GPU Memory: {recommendation['vram_peak_mb']:.1f} MB")
        
        # Apply configurations if requested
        if args.apply:
            apply_ok = suite.apply_configuration(recommendation)
            if apply_ok:
                print("  ✅ Recommended settings successfully applied to config/settings.yaml!")
            else:
                print("  ❌ Failed to write recommended settings to config/settings.yaml.")
        else:
            print("\n  👉 Pass the '--apply' flag to automatically configure your pipeline to use these optimal settings.")
            
    if report_ok:
        print(f"\n📊 Interactive HTML report generated at:")
        print(f"   file:///{os.path.abspath(report_path).replace(os.sep, '/')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
