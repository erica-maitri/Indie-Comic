"""
MANGAFLOW LAYOUT ENGINE — Phase 7
=================================
Arranges panels on pages dynamically based on action intensity, 
emotion pacing, and pacing metadata. Replaces the fixed 2x2 grid.
Implements advanced typesetting with gutters, page borders, and
focal crop fitting.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("pipeline.layout_engine")


class MangaFlowLayoutEngine:
    """
    Phase 7: MangaFlow Layout Engine.
    
    Replaces static grid layouts with dynamic, pacing-aware panel layouts.
    Allocates canvas space based on panel action intensity and size classes,
    applies professional gutters, and adds page numbering/borders.
    """

    def __init__(self, page_width: int = 1000, 
                 page_height: int = 1500,
                 gutter_width: int = 12,
                 margin: int = 40,
                 bg_color: str = "white"):
        self.page_width = page_width
        self.page_height = page_height
        self.gutter_width = gutter_width
        self.margin = margin
        self.bg_color = bg_color

    def layout_page(self, panels: List[Any], page_num: int) -> Image.Image:
        """
        Dynamically layouts panel images onto a single comic page.
        
        Args:
            panels: List of dicts or PIL Images representing panels on this page.
            page_num: Page number for typesetting
            
        Returns:
            Assembled PIL Image of the page
        """
        if not panels:
            # Empty page
            return Image.new("RGB", (self.page_width, self.page_height), self.bg_color)
            
        # Standardize panels: handle dicts, PIL Images, or other objects with image attribute
        standardized_panels = []
        for p in panels:
            if isinstance(p, dict):
                standardized_panels.append(p)
            elif isinstance(p, Image.Image):
                standardized_panels.append({"image": p, "action_intensity": 0.5})
            else:
                img = getattr(p, "image", None)
                if img is not None:
                    standardized_panels.append({"image": img, "action_intensity": getattr(p, "action_intensity", 0.5)})
                else:
                    standardized_panels.append({"image": p, "action_intensity": 0.5})
                    
        n = len(standardized_panels)
        log.info(f"MangaFlow Layout: Assembling Page {page_num} with {n} panels")
        
        # Create blank page canvas
        page = Image.new("RGB", (self.page_width, self.page_height), self.bg_color)
        draw = ImageDraw.Draw(page)
        
        # Calculate panel boxes
        boxes = self._calculate_panel_boxes(standardized_panels)
        
        # Place panels
        for i, panel in enumerate(standardized_panels):
            if i >= len(boxes):
                break
                
            panel_img = panel.get("image")
            if panel_img is None:
                continue
                
            box = boxes[i]  # (x, y, w, h)
            bx, by, bw, bh = box
            
            # Resize and crop panel image to fit the box aspect ratio
            fitted_img = self._fit_image_to_box(panel_img, bw, bh)
            
            # Paste onto page
            page.paste(fitted_img, (bx, by))
            
            # Draw a subtle panel outline
            draw.rectangle([bx, by, bx + bw, by + bh], outline=(40, 40, 40), width=3)
            
        # Draw page borders/frame (optional visual polish)
        draw.rectangle(
            [self.margin // 2, self.margin // 2, 
             self.page_width - self.margin // 2, self.page_height - self.margin // 2],
            outline=(180, 180, 180), width=1
        )
        
        # Add page numbering
        self._add_page_number(page, page_num)
        
        return page

    def _calculate_panel_boxes(self, panels: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """
        Compute bounding boxes for each panel on the page.
        Returns a list of (x, y, width, height) coordinates.
        """
        n = len(panels)
        content_w = self.page_width - 2 * self.margin
        content_h = self.page_height - 2 * self.margin
        
        boxes = []
        
        # Scenario 1: Single Panel (Full-Page Spread)
        if n == 1:
            boxes.append((self.margin, self.margin, content_w, content_h))
            
        # Scenario 2: Two Panels
        elif n == 2:
            # Split vertically (top/bottom) based on action intensity weights
            w1 = self._get_weight(panels[0])
            w2 = self._get_weight(panels[1])
            total_w = w1 + w2
            
            h1 = int(content_h * (w1 / total_w)) - self.gutter_width // 2
            h2 = content_h - h1 - self.gutter_width
            
            boxes.append((self.margin, self.margin, content_w, h1))
            boxes.append((self.margin, self.margin + h1 + self.gutter_width, content_w, h2))
            
        # Scenario 3: Three Panels
        elif n == 3:
            # Layout: Row 1 has 1 panel, Row 2 has 2 side-by-side panels
            w1 = self._get_weight(panels[0])
            w23 = (self._get_weight(panels[1]) + self._get_weight(panels[2])) / 2
            total_h_weight = w1 + w23
            
            h1 = int(content_h * (w1 / total_h_weight)) - self.gutter_width // 2
            h2 = content_h - h1 - self.gutter_width
            
            # Row 1 (Full width)
            boxes.append((self.margin, self.margin, content_w, h1))
            
            # Row 2 (Split width)
            w_left = self._get_weight(panels[1])
            w_right = self._get_weight(panels[2])
            total_row_w = w_left + w_right
            
            lw = int(content_w * (w_left / total_row_w)) - self.gutter_width // 2
            rw = content_w - lw - self.gutter_width
            
            boxes.append((self.margin, self.margin + h1 + self.gutter_width, lw, h2))
            boxes.append((self.margin + lw + self.gutter_width, self.margin + h1 + self.gutter_width, rw, h2))
            
        # Scenario 4: Four Panels (Pacing-aware Grid)
        elif n == 4:
            # Check if there is one dominant high-action panel
            weights = [self._get_weight(p) for p in panels]
            max_idx = weights.index(max(weights))
            
            if weights[max_idx] > 1.4:
                # Layout: Dominant panel takes 55% height, others share remainder
                h_dom = int(content_h * 0.55) - self.gutter_width // 2
                h_rest = content_h - h_dom - self.gutter_width
                
                if max_idx in (0, 1):
                    # Dominant row is at the top
                    if max_idx == 0:
                        boxes.append((self.margin, self.margin, content_w, h_dom))
                        # Row 2 (Three panels split side-by-side)
                        pw1 = (content_w - 2 * self.gutter_width) // 3
                        pw2 = pw1
                        pw3 = content_w - 2 * self.gutter_width - pw1 - pw2
                        boxes.append((self.margin, self.margin + h_dom + self.gutter_width, pw1, h_rest))
                        boxes.append((self.margin + pw1 + self.gutter_width, self.margin + h_dom + self.gutter_width, pw2, h_rest))
                        boxes.append((self.margin + pw1 + pw2 + 2 * self.gutter_width, self.margin + h_dom + self.gutter_width, pw3, h_rest))
                    else:
                        # Row 1 is dominant, but panel 2 is dominant (index 1).
                        self._fill_grid_layout(content_w, content_h, boxes)
                else:
                    # Dominant row is at the bottom (max_idx is 2 or 3)
                    pw1 = (content_w - 2 * self.gutter_width) // 3
                    pw2 = pw1
                    pw3 = content_w - 2 * self.gutter_width - pw1 - pw2
                    
                    if max_idx == 2:
                        boxes.append((self.margin, self.margin, pw1, h_rest))
                        boxes.append((self.margin + pw1 + self.gutter_width, self.margin, pw2, h_rest))
                        boxes.append((self.margin, self.margin + h_rest + self.gutter_width, content_w, h_dom))
                        boxes.append((self.margin + pw1 + pw2 + 2 * self.gutter_width, self.margin, pw3, h_rest))
                    else:  # max_idx == 3
                        boxes.append((self.margin, self.margin, pw1, h_rest))
                        boxes.append((self.margin + pw1 + self.gutter_width, self.margin, pw2, h_rest))
                        boxes.append((self.margin + pw1 + pw2 + 2 * self.gutter_width, self.margin, pw3, h_rest))
                        boxes.append((self.margin, self.margin + h_rest + self.gutter_width, content_w, h_dom))
            else:
                # Default: standard grid with slight offsets
                self._fill_grid_layout(content_w, content_h, boxes)
                
        # Scenario 5: Five or more Panels (Three-tier layout)
        else:
            # Tier 1 (2 panels) | Tier 2 (1 panel) | Tier 3 (2 panels)
            t1 = content_h // 3 - self.gutter_width
            t2 = content_h // 3 - self.gutter_width
            t3 = content_h - t1 - t2 - 2 * self.gutter_width
            
            # Row 1
            pw1 = content_w // 2 - self.gutter_width // 2
            boxes.append((self.margin, self.margin, pw1, t1))
            boxes.append((self.margin + pw1 + self.gutter_width, self.margin, pw1, t1))
            
            # Row 2
            boxes.append((self.margin, self.margin + t1 + self.gutter_width, content_w, t2))
            
            # Row 3
            pw3 = content_w // 2 - self.gutter_width // 2
            boxes.append((self.margin, self.margin + t1 + t2 + 2 * self.gutter_width, pw3, t3))
            boxes.append((self.margin + pw3 + self.gutter_width, self.margin + t1 + t2 + 2 * self.gutter_width, pw3, t3))
            
        return boxes

    def _fill_grid_layout(self, content_w: int, content_h: int, boxes: list):
        """Helper to create a standard 2x2 grid layout."""
        pw1 = content_w // 2 - self.gutter_width // 2
        pw2 = content_w - pw1 - self.gutter_width
        
        ph1 = content_h // 2 - self.gutter_width // 2
        ph2 = content_h - ph1 - self.gutter_width
        
        boxes.append((self.margin, self.margin, pw1, ph1))
        boxes.append((self.margin + pw1 + self.gutter_width, self.margin, pw2, ph1))
        boxes.append((self.margin, self.margin + ph1 + self.gutter_width, pw1, ph2))
        boxes.append((self.margin + pw1 + self.gutter_width, self.margin + ph1 + self.gutter_width, pw2, ph2))

    def _get_weight(self, panel: Dict[str, Any]) -> float:
        """Extract size weight based on action intensity."""
        intensity = panel.get("action_intensity", 0.5)
        # Convert intensity (0 to 1) to a scale factor (0.7 to 1.7)
        return 0.7 + intensity * 1.0

    def _fit_image_to_box(self, img: Image.Image, bw: int, bh: int) -> Image.Image:
        """Resize and crop an image to fit a specific box dimension exactly (focal crop)."""
        img_w, img_h = img.size
        
        # Calculate aspect ratios
        img_aspect = img_w / img_h
        box_aspect = bw / bh
        
        if img_aspect > box_aspect:
            # Image is too wide, fit height and crop sides
            new_h = bh
            new_w = int(bh * img_w // img_h)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Crop horizontal center
            start_x = (new_w - bw) // 2
            return resized.crop((start_x, 0, start_x + bw, bh))
        else:
            # Image is too tall, fit width and crop top/bottom
            new_w = bw
            new_h = int(bw * img_h // img_w)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Crop vertical center
            start_y = (new_h - bh) // 2
            return resized.crop((0, start_y, bw, start_y + bh))

    def _add_page_number(self, page: Image.Image, page_num: int):
        """Draw page number centered at the bottom of the page."""
        draw = ImageDraw.Draw(page)
        
        # Load a small font for page number
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            
        text = f"— Page {page_num} —"
        
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(text) * 8
            text_h = 16
            
        x = (self.page_width - text_w) // 2
        y = self.page_height - self.margin // 2 - text_h // 2
        
        # Draw white backdrop pill
        draw.rounded_rectangle(
            [x - 10, y - 4, x + text_w + 10, y + text_h + 4],
            radius=4,
            fill="white"
        )
        
        # Draw page number text
        draw.text((x, y), text, fill=(100, 100, 100), font=font)
