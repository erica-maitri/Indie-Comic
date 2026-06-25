import os
import zipfile
from typing import Optional

class ComicExporter:
    """Exports generated comics into standard reader formats like CBZ and CBR"""
    
    def __init__(self, output_dir="outputs/exports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_cbz(self, pages: list, title: str = "Comic") -> Optional[str]:
        """
        Export to CBZ (ZIP archive of images)
        This is the most universally supported comic book format.
        """
        # Format filename
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        safe_title = safe_title.replace(" ", "_")
        output_path = os.path.join(self.output_dir, f"{safe_title}.cbz")
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                for i, page in enumerate(pages):
                    # Expects page to be a dictionary containing 'page_image' (PIL Image)
                    img = page.get('page_image')
                    if img:
                        # Save image to temporary path
                        temp_path = f"temp_page_{i+1:03d}.png"
                        img.save(temp_path, "PNG")
                        
                        # Add to archive
                        cbz.write(temp_path, f"page_{i+1:03d}.png")
                        
                        # Cleanup temp
                        os.remove(temp_path)
            
            print(f"[OK] Successfully exported CBZ: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[!] Failed to export CBZ: {e}")
            return None
    
    def export_cbr(self, pages: list, title: str = "Comic") -> Optional[str]:
        """
        Export to CBR (RAR archive of images)
        Requires system-level RAR utility.
        If unavailable, falls back to standard CBZ output to prevent header corruption.
        """
        import shutil
        import subprocess

        # Find rar utility
        rar_path = shutil.which("rar") or shutil.which("rar.exe")
        if not rar_path:
            # Check common Windows paths
            win_rar = r"C:\Program Files\WinRAR\rar.exe"
            if os.path.exists(win_rar):
                rar_path = win_rar
                
        if not rar_path:
            print("[WARNING] Native 'rar' executable not found in PATH or standard location. Cannot create valid RAR-based CBR. Falling back to exporting a standard CBZ file instead.")
            return self.export_cbz(pages, title)
            
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        safe_title = safe_title.replace(" ", "_")
        output_path = os.path.join(self.output_dir, f"{safe_title}.cbr")
        
        # Temp directory for files to pack
        temp_files = []
        try:
            for i, page in enumerate(pages):
                img = page.get('page_image')
                if img:
                    temp_path = f"temp_page_{i+1:03d}.png"
                    img.save(temp_path, "PNG")
                    temp_files.append(temp_path)
            
            if not temp_files:
                print("[!] No page images to archive.")
                return None
                
            # Run RAR command: rar a -ep <output_path> <temp_files>
            # -ep excludes paths from names (store only filenames)
            # We delete the archive first if it exists to avoid appending
            if os.path.exists(output_path):
                os.remove(output_path)
                
            cmd = [rar_path, "a", "-ep", output_path] + temp_files
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                print(f"[OK] Successfully exported CBR: {output_path}")
                return output_path
            else:
                print(f"[!] RAR execution failed (return code {result.returncode}): {result.stderr or result.stdout}")
                print("[!] Falling back to CBZ.")
                return self.export_cbz(pages, title)
                
        except Exception as e:
            print(f"[!] Failed to export CBR: {e}. Falling back to CBZ.")
            return self.export_cbz(pages, title)
        finally:
            # Cleanup temp files
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    def export_web_comic(self, pages: list, output_path: str = "outputs/exports/web_comic.html"):
        """Export as web-comic HTML for scrolling"""
        # A simple vertical scrolling HTML format (Webtoon style)
        html = ["<html><head><title>Web Comic</title><style>body{background:#000;text-align:center;margin:0;} img{max-width:800px;width:100%;display:block;margin:0 auto;}</style></head><body>"]
        
        for i, page in enumerate(pages):
            # Save images locally relative to the html file
            img = page.get('page_image')
            if img:
                img_name = f"web_page_{i}.png"
                img_path = os.path.join(os.path.dirname(output_path), img_name)
                img.save(img_path, "PNG")
                html.append(f'<img src="{img_name}" />')
                
        html.append("</body></html>")
        
        with open(output_path, "w") as f:
            f.write("\n".join(html))
            
        return output_path
