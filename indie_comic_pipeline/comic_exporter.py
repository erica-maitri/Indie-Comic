import os
import zipfile
from typing import Optional, List

class ComicExporter:
    """Exports generated comics into standard reader formats like CBZ, CBR, PDF, and scrollable Web HTML."""
    
    def __init__(self, output_dir="outputs/exports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_cbz(self, pages: list, title: str = "Comic") -> Optional[str]:
        """
        Export to CBZ (ZIP archive of images with metadata.xml)
        This is the most universally supported comic book format.
        """
        # Format filename
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        safe_title = safe_title.replace(" ", "_")
        output_path = os.path.join(self.output_dir, f"{safe_title}.cbz")
        
        # Create metadata XML content
        metadata = f"""<?xml version="1.0" encoding="utf-8"?>
<ComicMetadata>
  <Title>{title}</Title>
  <PageCount>{len(pages)}</PageCount>
  <Creator>AI Indie Comic Generator</Creator>
  <Description>Generated comic book using AI Indie Comic Pipeline.</Description>
</ComicMetadata>
"""
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                # Add metadata.xml
                cbz.writestr("metadata.xml", metadata)
                
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

    def export_pdf(self, pages: list, title: str = "Comic") -> Optional[str]:
        """
        Export comic pages to PDF format.
        Uses reportlab if available, otherwise falls back to PIL direct PDF export.
        """
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        safe_title = safe_title.replace(" ", "_")
        output_path = os.path.join(self.output_dir, f"{safe_title}.pdf")
        
        try:
            # Try reportlab first
            from reportlab.pdfgen import canvas
            
            # Setup canvas
            c = canvas.Canvas(output_path)
            c.setTitle(title)
            
            for i, page in enumerate(pages):
                img = page.get('page_image')
                if img:
                    # Save a temp image to draw
                    temp_path = f"temp_pdf_page_{i+1:03d}.png"
                    img.save(temp_path, "PNG")
                    
                    # Page dimensions (using raw pixel size as point dimensions)
                    img_w, img_h = img.size
                    c.setPageSize((img_w, img_h))
                    c.drawImage(temp_path, 0, 0, width=img_w, height=img_h)
                    c.showPage()
                    
                    # Cleanup
                    os.remove(temp_path)
            c.save()
            print(f"[OK] Successfully exported PDF using reportlab: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[WARNING] reportlab PDF generation failed: {e}. Falling back to standard PIL PDF output.")
            try:
                # PIL fallback
                pil_pages = []
                for p in pages:
                    img = p.get('page_image')
                    if img:
                        pil_pages.append(img.convert('RGB'))
                if pil_pages:
                    pil_pages[0].save(
                        output_path,
                        save_all=True,
                        append_images=pil_pages[1:],
                        optimize=True,
                        quality=85
                    )
                    print(f"[OK] Successfully exported PDF using PIL fallback: {output_path}")
                    return output_path
            except Exception as e_pil:
                print(f"[!] Failed to export PDF with PIL fallback: {e_pil}")
            return None

    def export_web_comic(self, pages: list, output_path: str = "outputs/exports/web_comic.html"):
        """Export as web-comic HTML for scrolling with clean styling"""
        title = "AI Indie Comic"
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"  <title>{title}</title>",
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <style>",
            "    body {",
            "      background-color: #0c0d12;",
            "      color: #e2e8f0;",
            "      font-family: 'Inter', system-ui, sans-serif;",
            "      margin: 0;",
            "      padding: 0;",
            "      display: flex;",
            "      flex-direction: column;",
            "      align-items: center;",
            "    }",
            "    header {",
            "      width: 100%;",
            "      background: rgba(15, 17, 26, 0.95);",
            "      border-bottom: 1px solid #1e293b;",
            "      padding: 15px 0;",
            "      text-align: center;",
            "      position: sticky;",
            "      top: 0;",
            "      z-index: 100;",
            "      backdrop-filter: blur(10px);",
            "    }",
            "    h1 {",
            "      margin: 0;",
            "      font-size: 1.5rem;",
            "      font-weight: 700;",
            "      letter-spacing: -0.025em;",
            "      color: #f8fafc;",
            "    }",
            "    .comic-container {",
            "      max-width: 800px;",
            "      width: 100%;",
            "      margin: 20px auto 40px auto;",
            "      display: flex;",
            "      flex-direction: column;",
            "      gap: 16px;",
            "      padding: 0 16px;",
            "      box-sizing: border-box;",
            "    }",
            "    .page-wrapper {",
            "      background: #0f111a;",
            "      border: 1px solid #1e293b;",
            "      border-radius: 8px;",
            "      overflow: hidden;",
            "      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);",
            "      transition: transform 0.2s;",
            "    }",
            "    .page-wrapper:hover {",
            "      transform: scale(1.005);",
            "    }",
            "    img {",
            "      width: 100%;",
            "      display: block;",
            "      height: auto;",
            "    }",
            "    .page-footer {",
            "      padding: 8px 12px;",
            "      background: #1e293b;",
            "      font-size: 0.75rem;",
            "      text-align: right;",
            "      color: #94a3b8;",
            "    }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <header><h1>{title}</h1></header>",
            '  <div class="comic-container">'
        ]
        
        for i, page in enumerate(pages):
            img = page.get('page_image')
            if img:
                img_name = f"web_page_{i+1:03d}.png"
                img_path = os.path.join(os.path.dirname(output_path), img_name)
                img.save(img_path, "PNG")
                html.append(f'    <div class="page-wrapper">')
                html.append(f'      <img src="{img_name}" alt="Page {i+1}" />')
                html.append(f'      <div class="page-footer">Page {i+1} of {len(pages)}</div>')
                html.append(f'    </div>')
                
        html.append("  </div>")
        html.append("</body>")
        html.append("</html>")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
            
        return output_path
