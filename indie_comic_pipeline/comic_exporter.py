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
        Requires a system-level RAR utility (like WinRAR) or the rarfile library if RAR is in PATH.
        Usually, CBZ is preferred because zip is natively supported in Python.
        """
        try:
            import rarfile  # type: ignore
            # Note: Creating RAR files typically requires an external executable.
            # rarfile library is primarily for reading. 
            # If the user absolutely needs CBR, we would shell out to 'rar a'.
            # We will fallback to CBZ internally but rename the extension for readers that just inspect headers.
            print("[i] CBR export requested. Falling back to CBZ creation with .cbr extension for compatibility.")
            
            output_cbz = self.export_cbz(pages, title)
            if output_cbz:
                output_cbr = output_cbz.replace(".cbz", ".cbr")
                os.rename(output_cbz, output_cbr)
                return output_cbr
            return None
                
        except (ImportError, Exception) as e:
            print(f"[!] Cannot process native CBR ({e}). Exporting CBZ instead.")
            return self.export_cbz(pages, title)

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
