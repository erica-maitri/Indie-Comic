"""
TEXT-IMAGE INTEGRATOR — Phase 5 (DiffSensei Approximation)
============================================================
Implements the DiffSensei architectural slot with a best-effort
approximation for integrated text-image generation:
- Takes raw panel raster + script dialogue
- Connects to local Ollama to plan optimal text/bubble placement
- Saves and loads placement settings to/from local JSON files
- Applies emotion-aware speech bubble styling
- Renders dialogue with dynamic expression-matched typography
- Positions bubbles using layout ratios or speaker coordinates
"""

import os
import math
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

log = logging.getLogger("pipeline.text_image")


# Emotion-specific bubble style configurations
BUBBLE_STYLES = {
    # Beat category → (shape, border_width, fill_color, border_color, font_scale)
    "calm": {
        "shape": "ellipse", "border_width": 2, "fill": (255, 255, 255, 230),
        "border_color": (40, 40, 40), "font_scale": 1.0, "tail_style": "smooth",
    },
    "intense": {
        "shape": "jagged", "border_width": 3, "fill": (255, 255, 240, 240),
        "border_color": (180, 30, 30), "font_scale": 1.15, "tail_style": "sharp",
    },
    "thought": {
        "shape": "cloud", "border_width": 2, "fill": (240, 240, 255, 200),
        "border_color": (100, 100, 140), "font_scale": 0.9, "tail_style": "bubbles",
    },
    "whisper": {
        "shape": "dashed_ellipse", "border_width": 1, "fill": (255, 255, 255, 180),
        "border_color": (120, 120, 120), "font_scale": 0.85, "tail_style": "smooth",
    },
    "shout": {
        "shape": "spiky", "border_width": 4, "fill": (255, 250, 230, 245),
        "border_color": (200, 50, 20), "font_scale": 1.3, "tail_style": "sharp",
    },
}

# Map emotion beats to bubble categories
BEAT_TO_BUBBLE = {
    # Sad / Grief arc
    "heaviness": "whisper",
    "stillness": "whisper",
    "faint_warmth": "calm",
    "tentative_light": "thought",
    "soft_openness": "calm",
    "quiet_hope": "calm",
    # Angry arc
    "contained_fire": "intense",
    "fracture": "intense",
    "exhale": "intense",
    "cooling": "intense",
    "ground": "intense",
    # Tired / Exhausted arc
    "drag": "intense",
    "surrender": "intense",
    "softness": "calm",
    "drift": "thought",
    "quiet_rest": "whisper",
    "renewal": "shout",
    # Happy / Elation arc
    "spark": "shout",
    "expansion": "shout",
    "overflow": "shout",
    "radiance": "shout",
    "luminous_still": "shout",
    "transcendence": "shout",
    # Anxious arc
    "spiral": "intense",
    "peak_noise": "shout",
    "pause": "intense",
    "breath": "intense",
    "root": "intense",
    "present": "calm",
    # Grief arc
    "absence": "thought",
    "ache": "whisper",
    "memory": "thought",
    "held": "thought",
    "continuance": "shout",
    "carried_forward": "shout",
    # Determined arc
    "doubt": "intense",
    "challenge": "intense",
    "resistance": "intense",
    "breakthrough": "shout",
    "momentum": "intense",
    "triumph": "shout",
    # Love arc
    "recognition": "calm",
    "vulnerability": "thought",
    "trust": "calm",
    "embrace": "shout",
    "unity": "shout",
    # Generic / Fallbacks
    "neutral": "calm",
    "resolution": "calm",
    "acknowledgment": "calm",
    "presence": "calm",
    "shift": "thought",
    "openness": "calm",
}


class TextImageIntegrator:
    """
    Phase 5: DiffSensei Approximation.

    Integrates dialogue text into generated panel images with
    emotion-aware speech bubble styling and dynamic typography.
    Uses local Ollama for smart layout planning and caches/loads configurations
    from local JSON layout files.
    """

    def __init__(self, font_path: Optional[str] = None,
                 base_font_size: int = 16,
                 max_bubble_width_ratio: float = 0.45,
                 output_dir: str = "outputs/panels",
                 ollama_model: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434"):
        self.font_path = font_path
        self.base_font_size = base_font_size
        self.max_bubble_width_ratio = max_bubble_width_ratio
        self.output_dir = output_dir
        self.ollama_model = ollama_model
        self.ollama_url = os.environ.get("OLLAMA_URL") or ollama_url
        self._font_cache: Dict[Tuple[int, str], Any] = {}
        self._llm = None
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._download_fonts_if_missing()

    def _download_fonts_if_missing(self):
        """Download Comic Neue (Regular, Bold, Italic) TTF files if they do not exist locally."""
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "fonts")
        os.makedirs(font_dir, exist_ok=True)
        
        urls = {
            "regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Regular.ttf",
            "bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf",
            "italic": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Italic.ttf"
        }
        
        import urllib.request
        for style, url in urls.items():
            out_path = os.path.join(font_dir, f"ComicNeue-{style.capitalize()}.ttf")
            if not os.path.exists(out_path):
                log.info(f"Downloading Comic Neue {style} font to {out_path}...")
                try:
                    urllib.request.urlretrieve(url, out_path)
                    log.info(f"Successfully downloaded Comic Neue {style} font.")
                except Exception as e:
                    log.warning(f"Failed to download Comic Neue {style} font: {e}")

    def _get_llm(self):
        """Lazy-load the appropriate LLM connection based on provider configuration."""
        if self._llm is None:
            provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
            log.info(f"[TextImageIntegrator] Initializing LLM provider: {provider}")
            
            try:
                if provider == "openai":
                    from langchain_openai import ChatOpenAI  # type: ignore
                    self._llm = ChatOpenAI(
                        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        temperature=0.1
                    )
                    log.info(f"Connected to OpenAI for TextImageIntegrator: {self._llm.model_name}")
                elif provider == "gemini":
                    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                    self._llm = ChatGoogleGenerativeAI(
                        model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
                        temperature=0.1
                    )
                    log.info(f"Connected to Gemini for TextImageIntegrator: {self._llm.model}")
                elif provider == "anthropic":
                    from langchain_anthropic import ChatAnthropic  # type: ignore
                    self._llm = ChatAnthropic(
                        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                        temperature=0.1
                    )
                    log.info(f"Connected to Anthropic for TextImageIntegrator: {self._llm.model}")
                else:
                    # Default: Ollama
                    from langchain_ollama import ChatOllama
                    self._llm = ChatOllama(
                        model=self.ollama_model,
                        temperature=0.1,
                        base_url=self.ollama_url,
                    )
                    log.info(f"Connected to Ollama for TextImageIntegrator: {self.ollama_model}")
            except Exception as e:
                log.warning(f"Failed to load LLM provider '{provider}' in TextImageIntegrator: {e}. Using manual/json fallback.")
                self._llm = None
        return self._llm

    def _call_ollama_api(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Make a direct HTTP request to Ollama's local server as a robust fallback/default."""
        import urllib.request
        import urllib.error
        
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                return resp_data.get("response", "")
        except Exception as e:
            log.warning(f"Direct Ollama HTTP request failed: {e}")
            return None

    def _parse_ollama_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse a JSON object from text."""
        try:
            clean = re.sub(r"^```(?:json)?\s*", "", text.strip())
            clean = re.sub(r"\s*```$", "", clean).strip()
            
            start = clean.find("{")
            if start == -1:
                return None
                
            depth, end = 0, -1
            for i, ch in enumerate(clean[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                return None
                
            return json.loads(clean[start:end + 1])
        except Exception as e:
            log.warning(f"Error parsing Ollama JSON: {e}")
            return None

    def get_layout_plan(self, dialogue: str,
                        emotion_beat: str = "neutral",
                        panel_id: int = 0,
                        scene_desc: Optional[str] = None,
                        speaker_position: str = "center") -> Dict[str, Any]:
        """
        Get bubble placement and styling plan using Ollama and local JSON files.
        
        Checks if a local layout JSON exists first. If so, loads it.
        Otherwise, asks local Ollama to plan it and saves the plan to local JSON.
        """
        json_filename = f"panel_{panel_id:03d}_bubble_layout.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        # 2. Parse speaker and basic dialogue to check against cache
        speaker, text_clean = self._parse_dialogue(dialogue)
        if not text_clean:
            text_clean = dialogue

        # 1. Load from local JSON if it exists (allows user editing of placements!)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    plan = json.load(f)
                
                # Check if cached dialogue matches the current dialogue text
                plan_dialogue = plan.get("dialogue_clean", "")
                if plan_dialogue.strip().lower() == text_clean.strip().lower():
                    log.info(f"Loaded matching bubble layout plan from local JSON: {json_path}")
                    return plan
                else:
                    log.info(f"Dialogue changed for panel {panel_id} — ignoring cached layout JSON")
            except Exception as e:
                log.warning(f"Failed to read local layout JSON {json_path}: {e}")
        bubble_cat = BEAT_TO_BUBBLE.get(emotion_beat, "calm")
        style = BUBBLE_STYLES.get(bubble_cat, BUBBLE_STYLES["calm"])
        
        # Heuristic positioning to avoid overlays on faces
        if not speaker_position or speaker_position == "center":
            val = panel_id
            if speaker:
                val += sum(ord(c) for c in speaker)
            pos_options = ["left", "right", "center"]
            speaker_pos = pos_options[val % len(pos_options)]
        else:
            speaker_pos = speaker_position
            
        x_ratio = 0.5
        if speaker_pos == "left":
            x_ratio = 0.25
        elif speaker_pos == "right":
            x_ratio = 0.75
            
        y_ratio = 0.15 + ((panel_id * 7) % 3) * 0.08
        
        plan = {
            "speaker": speaker,
            "dialogue_clean": text_clean,
            "bubble_shape": style.get("shape", "ellipse"),
            "speaker_position": speaker_pos,
            "font_scale": style.get("font_scale", 1.0),
            "x_ratio": x_ratio,
            "y_ratio": y_ratio,
            "text_align": "center",
            "tail_x_ratio": None,
            "tail_y_ratio": None,
            "source": "heuristic_fallback"
        }
        
        # 3. Call local Ollama
        system_prompt = """You are a comic book lettering coordinator.
Analyze the dialogue, emotion beat, and panel details, then output a JSON object planning the speech bubble layout.
Respond ONLY with a JSON object. No explanation. No markdown formatting.

Available bubble shapes:
- "ellipse" (standard dialogue)
- "dashed_ellipse" (whispers, quiet thoughts)
- "jagged" (angry, tense, stressed)
- "cloud" (thought bubble, thinking)
- "spiky" (screaming, shouting)

Relative positions:
- "left" (speaker is on the left of the image)
- "right" (speaker is on the right of the image)
- "center" (speaker is in the middle of the image)

Text alignment:
- "center" (standard comic book alignment)
- "left" (for narratives or notebooks)
- "right" (for sound effects/special placement)

JSON structure:
{
  "speaker": "name of speaker or null",
  "dialogue_clean": "dialogue text without character name prefix. Optionally add markdown emphasis on IMPORTANT words only (e.g. **bold** for major stress/shouting, *italic* for quiet stress/thoughts). NEVER bold random words.",
  "bubble_shape": "ellipse|dashed_ellipse|jagged|cloud|spiky",
  "speaker_position": "left|center|right",
  "font_scale": 1.0,
  "x_ratio": 0.5,
  "y_ratio": 0.15,
  "text_align": "center|left|right",
  "tail_x_ratio": 0.5,
  "tail_y_ratio": 0.8
}"""

        prompt = f"""Panel ID: {panel_id}
Dialogue: "{dialogue}"
Emotion Beat: {emotion_beat}
Scene/Action Description: "{scene_desc or 'Not specified'}"

Please design the speech bubble layout. Determine the speaker, clean dialogue, bubble shape best suited to the emotional beat, relative horizontal position (left, right, or center) and the relative X (x_ratio) and Y (y_ratio) coordinates for bubble placement (ratios between 0.05 and 0.95). Choose a suitable text_align (usually "center" for bubbles). Predict the coordinates of the speaker's head/body as tail_x_ratio and tail_y_ratio so the speech bubble tail points towards them. Keep the bubble y_ratio near the top (e.g. 0.15 to 0.3) so it doesn't cover main character bodies, but adjust slightly based on panel ID to avoid overlap.
"""
        
        log.info(f"Querying local Ollama ({self.ollama_model}) for bubble layout on panel {panel_id}...")
        llm_response = self._call_ollama_api(prompt, system_prompt)
        
        if not llm_response:
            llm = self._get_llm()
            if llm:
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt),
                    ]
                    llm_response = llm.invoke(messages).content
                except Exception as e:
                    log.warning(f"langchain_ollama invoke failed in TextImageIntegrator: {e}")
                    
        if llm_response:
            ollama_plan = self._parse_ollama_json(llm_response)
            if ollama_plan:
                for key in ["speaker", "dialogue_clean", "bubble_shape", "speaker_position", "font_scale", "x_ratio", "y_ratio", "text_align", "tail_x_ratio", "tail_y_ratio"]:
                    if key in ollama_plan:
                        plan[key] = ollama_plan[key]
                plan["source"] = f"ollama_{self.ollama_model}"
                log.info(f"Ollama planned speech bubble: {plan}")
                
        # 4. Save to local JSON file for future use and editing
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2)
            log.info(f"Saved bubble layout plan to local JSON: {json_path}")
        except Exception as e:
            log.warning(f"Failed to save local layout JSON {json_path}: {e}")
            
        return plan

    def _parse_rich_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse a line of text containing Markdown-like tokens:
        **bold** or __bold__ -> Bold style
        *italic* or _italic_ -> Italic style
        """
        pattern = re.compile(r'(\*\*.*?\*\*|__.*?__|\*.*?\*|_.*?_|[^*_]+|[*_])')
        matches = pattern.findall(text)
        
        segments = []
        for m in matches:
            if not m:
                continue
            if (m.startswith("**") and m.endswith("**")) or (m.startswith("__") and m.endswith("__")):
                segments.append((m[2:-2], "bold"))
            elif (m.startswith("*") and m.endswith("*")) or (m.startswith("_") and m.endswith("_")):
                segments.append((m[1:-1], "italic"))
            else:
                segments.append((m, "regular"))
        return segments

    def _rich_text_width(self, text: str, font) -> int:
        """Measure the total width of a rich text string including style changes."""
        segments = self._parse_rich_text(text)
        total_w = 0
        size = font.size if hasattr(font, "size") else self.base_font_size
        for val, style in segments:
            style_font = self._get_font(size, style)
            total_w += self._text_width(val, style_font)
        return total_w

    def integrate(self, image: Image.Image,
                  dialogue: str,
                  emotion_beat: str = "neutral",
                  speaker_position: str = "center",
                  panel_id: int = 0,
                  scene_desc: Optional[str] = None) -> Image.Image:
        """
        Integrate dialogue into the panel image with styled speech bubbles.

        Args:
            image: Source panel PIL Image
            dialogue: Dialogue text (can include "Speaker: text" format)
            emotion_beat: Current emotional beat for styling
            speaker_position: Default horizontal position if plan lacks it
            panel_id: Panel identifier for deterministic layout
            scene_desc: Optional scene context for Ollama planning

        Returns:
            Image with integrated speech bubbles
        """
        if isinstance(dialogue, dict):
            speaker = dialogue.get("speaker") or dialogue.get("id") or dialogue.get("character")
            text = dialogue.get("text") or dialogue.get("dialogue") or ""
            if speaker and text:
                dialogue_str = f"{speaker}: {text}"
            elif text:
                dialogue_str = text
            else:
                found_str = ""
                for k, v in dialogue.items():
                    if isinstance(v, str):
                        found_str = v
                        break
                dialogue_str = found_str if found_str else str(dialogue)
            dialogue = dialogue_str

        if not dialogue or not isinstance(dialogue, str) or dialogue.strip() in ("...", ""):
            return image  # Silent panel

        # Get bubble layout plan (Ollama / Local JSON)
        plan = self.get_layout_plan(dialogue, emotion_beat, panel_id, scene_desc, speaker_position)
        
        # Extract variables from plan
        speaker = plan.get("speaker")
        text = plan.get("dialogue_clean", dialogue)
        if not text or text.strip() == "...":
            return image

        bubble_shape = plan.get("bubble_shape", "ellipse")
        speaker_pos = plan.get("speaker_position", speaker_position)
        font_scale = plan.get("font_scale", 1.0)
        x_ratio = plan.get("x_ratio", 0.5)
        y_ratio = plan.get("y_ratio", 0.15)
        text_align = plan.get("text_align", "center")

        # Get style from mapping or overrides
        bubble_cat = BEAT_TO_BUBBLE.get(emotion_beat, "calm")
        style = BUBBLE_STYLES.get(bubble_cat, BUBBLE_STYLES["calm"]).copy()
        
        # Override styling fields from the plan shape
        style["shape"] = bubble_shape
        style["font_scale"] = font_scale

        # Create a working copy
        result = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate bubble dimensions
        w, h = result.size
        max_bubble_w = int(w * self.max_bubble_width_ratio)
        font_scale_val = style.get("font_scale", 1.0)
        font_scale: float = 1.0
        if isinstance(font_scale_val, (int, float)):
            font_scale = float(font_scale_val)
        font_size = int(self.base_font_size * font_scale)
        font = self._get_font(font_size)

        # Use larger padding for cloud shapes to prevent circular arcs from clipping text
        padding = 20 if bubble_shape == "cloud" else 14

        # Wrap text to fit bubble width
        wrapped_lines = self._wrap_text(text, font, max_bubble_w - 20)

        # Calculate text block size
        line_height = font_size + 6  # Spacing for comic fonts
        text_height = 0
        max_line_w = 0
        
        for line in wrapped_lines:
            if line == "":
                text_height += 8  # paragraph break gap
            else:
                text_height += line_height
                lw = self._rich_text_width(line, font)
                if lw > max_line_w:
                    max_line_w = lw
                    
        text_width = max_line_w if max_line_w > 0 else 100

        # Bubble dimensions with padding
        bubble_w = text_width + padding * 2
        bubble_h = text_height + padding * 2

        # If speaker name, add extra height
        if speaker:
            bubble_h += line_height

        # Position the bubble using ratios
        bx, by = self._calculate_bubble_position(
            w, h, bubble_w, bubble_h, speaker_pos, panel_id, x_ratio, y_ratio, bubble_shape
        )

        # Draw the bubble shape
        self._draw_bubble(draw, bx, by, bubble_w, bubble_h, style)

        # Draw tail
        tail_x_ratio = plan.get("tail_x_ratio")
        tail_y_ratio = plan.get("tail_y_ratio")
        
        if tail_x_ratio is not None and tail_y_ratio is not None:
            target_x = int(tail_x_ratio * w)
            target_y = int(tail_y_ratio * h)
        else:
            target_x = self._get_tail_x(bx, bubble_w, speaker_pos, x_ratio)
            # Default thought tail is longer to accommodate lobes and bubble spacing
            tail_len = 45 if bubble_shape == "cloud" else 15
            target_y = by + bubble_h + tail_len
            
        base_x = self._get_tail_x(bx, bubble_w, speaker_pos, x_ratio)
        base_y = by + bubble_h
        
        if bubble_shape == "cloud":
            style["tail_style"] = "bubbles"
            
        self._draw_tail(draw, base_x, base_y, target_x, target_y, style)

        # Draw speaker name (bold) if present
        text_y = by + padding
        if speaker:
            speaker_font = self._get_font(int(font_size * 0.85), "bold")
            speaker_text = speaker.upper()
            speaker_w = self._text_width(speaker_text, speaker_font)
            speaker_x = bx + padding + (text_width - speaker_w) // 2 if text_align == "center" else (
                bx + padding + (text_width - speaker_w) if text_align == "right" else bx + padding
            )
            draw.text((speaker_x, text_y), speaker_text,
                      fill=(60, 60, 60, 255), font=speaker_font)
            text_y += line_height

        # Draw dialogue text
        for line in wrapped_lines:
            if line == "":
                text_y += 8  # Paragraph break gap
                continue
                
            line_w = self._rich_text_width(line, font)
            if text_align == "center":
                line_x = bx + padding + (text_width - line_w) // 2
            elif text_align == "right":
                line_x = bx + padding + (text_width - line_w)
            else:
                line_x = bx + padding

            # Draw segment by segment
            segments = self._parse_rich_text(line)
            cursor_x = line_x
            for val, style_var in segments:
                style_font = self._get_font(font_size, style_var)
                draw.text((cursor_x, text_y), val,
                          fill=(20, 20, 20, 255), font=style_font)
                cursor_x += self._text_width(val, style_font)
                
            text_y += line_height

        # Composite the overlay
        result = Image.alpha_composite(result, overlay)
        return result.convert("RGB")

    def integrate_batch(self, panels: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Integrate dialogue into multiple panels.

        Args:
            panels: List of dicts with keys: image, dialogue, emotion_beat, panel_id, scene_desc

        Returns:
            List of images with integrated text
        """
        results = []
        for panel in panels:
            img = self.integrate(
                image=panel["image"],
                dialogue=panel.get("dialogue", "..."),
                emotion_beat=panel.get("emotion_beat", "neutral"),
                speaker_position=panel.get("speaker_position", "center"),
                panel_id=panel.get("panel_id", 0),
                scene_desc=panel.get("scene_desc")
            )
            results.append(img)
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Bubble Drawing
    # ─────────────────────────────────────────────────────────────────────

    def _draw_bubble(self, draw: Any,
                     x: int, y: int, w: int, h: int,
                     style: Dict[str, Any]):
        """Draw the speech bubble shape."""
        shape = style.get("shape", "ellipse")
        fill = style.get("fill", (255, 255, 255, 230))
        border = style.get("border_color", (40, 40, 40))
        border_w = style.get("border_width", 2)

        if shape == "ellipse":
            radius = min(w, h) // 4
            self._draw_rounded_rect(draw, x, y, w, h, radius, fill, border, border_w)

        elif shape == "dashed_ellipse":
            radius = min(w, h) // 4
            self._draw_dashed_rounded_rect(draw, x, y, w, h, radius, fill, border, border_w)

        elif shape == "jagged":
            radius = 6
            self._draw_rounded_rect(draw, x, y, w, h, radius, fill, border, border_w)

        elif shape == "cloud":
            self._draw_cloud_bubble(draw, x, y, w, h, fill, border, border_w)

        elif shape == "spiky":
            self._draw_spiky_bubble(draw, x, y, w, h, fill, border, border_w)

        else:
            self._draw_rounded_rect(draw, x, y, w, h, 12, fill, border, border_w)

    def _draw_rounded_rect(self, draw: Any,
                           x: int, y: int, w: int, h: int,
                           radius: int, fill, outline, width: int):
        """Draw a rounded rectangle."""
        draw.rounded_rectangle(
            [x, y, x + w, y + h],
            radius=radius,
            fill=fill,
            outline=outline,
            width=width,
        )

    def _draw_dashed_rounded_rect(self, draw: Any, x: int, y: int, w: int, h: int, radius: int, fill: Any, border: Any, width: int):
        """Draw a rounded rectangle with a dashed outline."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
        
        step = 12
        dash_len = 6
        
        # Top edge
        for dx in range(x + radius, x + w - radius, step):
            draw.line([(dx, y), (min(dx + dash_len, x + w - radius), y)], fill=border, width=width)
        # Bottom edge
        for dx in range(x + radius, x + w - radius, step):
            draw.line([(dx, y + h), (min(dx + dash_len, x + w - radius), y + h)], fill=border, width=width)
        # Left edge
        for dy in range(y + radius, y + h - radius, step):
            draw.line([(x, dy), (x, min(dy + dash_len, y + h - radius))], fill=border, width=width)
        # Right edge
        for dy in range(y + radius, y + h - radius, step):
            draw.line([(x + w, dy), (x + w, min(dy + dash_len, y + h - radius))], fill=border, width=width)
            
        # Draw corner arcs
        draw.arc([x, y, x + 2*radius, y + 2*radius], 180, 270, fill=border, width=width)
        draw.arc([x + w - 2*radius, y, x + w, y + 2*radius], 270, 360, fill=border, width=width)
        draw.arc([x, y + h - 2*radius, x + 2*radius, y + h], 90, 180, fill=border, width=width)
        draw.arc([x + w - 2*radius, y + h - 2*radius, x + w, y + h], 0, 90, fill=border, width=width)

    def _draw_cloud_bubble(self, draw: Any, x: int, y: int, w: int, h: int, fill: Any, border: Any, border_w: int):
        """Draw a cloud thought bubble using the exact circle configuration of the user's generator."""
        cx = x + w / 2
        cy = y + h / 2
        
        core_w = w
        core_h = h
        
        # Calculate big and small radii relative to the core bubble height
        big_r = core_h * 0.45
        small_r = core_h * 0.38
        
        # Exact circle positions and scaling ratios from user code
        circles = [
            (cx - core_w * 0.20, ccy_val := cy - core_h * 0.15, big_r * 1.00),
            (cx + core_w * 0.18, ccy_val := cy - core_h * 0.18, big_r * 1.05),
            (cx - core_w * 0.42, ccy_val := cy - core_h * 0.00, big_r * 0.75),
            (cx + core_w * 0.40, ccy_val := cy + core_h * 0.05, big_r * 0.80),
            (cx - core_w * 0.30, ccy_val := cy + core_h * 0.20, small_r * 0.90),
            (cx + core_w * 0.02, ccy_val := cy + core_h * 0.22, small_r * 1.00),
            (cx + core_w * 0.30, ccy_val := cy + core_h * 0.18, small_r * 0.90),
            (cx - core_w * 0.05, ccy_val := cy - core_h * 0.02, small_r * 1.10),
        ]
        
        # Pass 1: Draw outer border outlines (filled with border color, radius enlarged by border_w)
        for (ccx, ccy, r) in circles:
            draw.ellipse([ccx - r - border_w, ccy - r - border_w, ccx + r + border_w, ccy + r + border_w], fill=border)
            
        # Pass 2: Draw inner fills (filled with fill color, exact radius r)
        for (ccx, ccy, r) in circles:
            draw.ellipse([ccx - r, ccy - r, ccx + r, ccy + r], fill=fill)

    def _draw_spiky_bubble(self, draw: Any, x: int, y: int, w: int, h: int, fill: Any, border: Any, border_w: int):
        """Draw a starburst shout bubble using the exact mathematical logic from the user's starburst generator."""
        import math
        cx = x + w / 2
        cy = y + h / 2
        rx = w / 2
        ry = h / 2
        
        points = []
        num_spikes = 12  # Exactly 12 spikes as configured in user code
        n_points = num_spikes * 2
        
        # Spike ratio (how far spikes stick out vs inner radius) is 1.55
        spike_ratio = 1.55
        
        for i in range(n_points):
            angle = math.pi * 2 * i / n_points - math.pi / 2
            
            # Alternate between outer and inner radius
            if i % 2 == 0:
                dx = rx * spike_ratio
                dy = ry * spike_ratio
            else:
                dx = rx
                dy = ry
                
            px = cx + dx * math.cos(angle)
            py = cy + dy * math.sin(angle)
            points.append((px, py))
            
        draw.polygon(points, fill=fill, outline=border, width=border_w)

    def _draw_tail(self, draw: Any,
                   base_x: int, base_y: int,
                   target_x: int, target_y: int,
                   style: Dict[str, Any]):
        """Draw the speech bubble tail pointing toward the target speaker coordinate."""
        tail_style = style.get("tail_style", "smooth")
        fill = style.get("fill", (255, 255, 255, 230))
        border = style.get("border_color", (40, 40, 40))

        tail_width = 12

        if tail_style == "bubbles":
            # Offset base_y down by 12px to clear the bulging cloud lobes
            adjusted_base_y = base_y + 12
            
            dx = target_x - base_x
            dy = target_y - adjusted_base_y
            D = math.sqrt(dx*dx + dy*dy)
            
            # Thought bubble circles with decreasing sizes (6px, 4px, 2px)
            radii = [6, 4, 2]
            
            if D < 35:
                # If short tail, space them out using relative ratios
                ratios = [0.3, 0.65, 0.9]
                for i in range(3):
                    t = ratios[i]
                    cx = int(base_x + dx * t)
                    cy = int(adjusted_base_y + dy * t)
                    r = radii[i]
                    draw.ellipse(
                        [cx - r, cy - r, cx + r, cy + r],
                        fill=fill, outline=border, width=1,
                    )
            else:
                # If long tail, place them at fixed, beautifully spaced pixel distances
                ux = dx / D
                uy = dy / D
                distances = [10, 22, 32]
                for i in range(3):
                    dist = distances[i]
                    cx = int(base_x + ux * dist)
                    cy = int(adjusted_base_y + uy * dist)
                    r = radii[i]
                    draw.ellipse(
                        [cx - r, cy - r, cx + r, cy + r],
                        fill=fill, outline=border, width=1,
                    )
        else:
            points = [
                (base_x - tail_width // 2, base_y),
                (base_x + tail_width // 2, base_y),
                (target_x, target_y),
            ]
            draw.polygon(points, fill=fill, outline=border)
            
            # Mask the bubble border at the tail base
            border_w = style.get("border_width", 2)
            draw.line([(base_x - tail_width // 2 + 1, base_y), (base_x + tail_width // 2 - 1, base_y)], fill=fill, width=border_w + 1)

    # ─────────────────────────────────────────────────────────────────────
    # Positioning
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_bubble_position(self, img_w: int, img_h: int,
                                   bubble_w: int, bubble_h: int,
                                   speaker_pos: str,
                                   panel_id: int,
                                   x_ratio: Optional[float] = None,
                                   y_ratio: Optional[float] = None,
                                   bubble_shape: str = "ellipse") -> Tuple[int, int]:
        """Calculate bubble position based on speaker position, panel ID, and optional planned ratios."""
        margin = 15
        outer_margin = 15 if bubble_shape in ("spiky", "cloud") else 0

        if x_ratio is not None and y_ratio is not None:
            # Centered on ratio coordinates
            x = int(x_ratio * img_w - bubble_w // 2)
            y = int(y_ratio * img_h - bubble_h // 2)
        else:
            if speaker_pos == "left":
                x = margin + outer_margin
            elif speaker_pos == "right":
                x = img_w - bubble_w - margin - outer_margin
            else:
                x = (img_w - bubble_w) // 2

            y_offset = (panel_id % 3) * 60
            y = margin + y_offset + outer_margin

        # Clamp to image bounds
        x = max(margin + outer_margin, min(x, img_w - bubble_w - margin - outer_margin))
        y = max(margin + outer_margin, min(y, img_h - bubble_h - 40 - outer_margin))

        return x, y

    def _get_tail_x(self, bubble_x: int, bubble_w: int,
                    speaker_pos: str, x_ratio: Optional[float] = None) -> int:
        """Get the x position for the bubble tail."""
        if speaker_pos == "left":
            return bubble_x + bubble_w // 4
        elif speaker_pos == "right":
            return bubble_x + 3 * bubble_w // 4
        else:
            if x_ratio is not None:
                if x_ratio < 0.4:
                    return bubble_x + bubble_w // 4
                elif x_ratio > 0.6:
                    return bubble_x + 3 * bubble_w // 4
            return bubble_x + bubble_w // 2

    # ─────────────────────────────────────────────────────────────────────
    # Text Utilities
    # ─────────────────────────────────────────────────────────────────────

    def _parse_dialogue(self, dialogue: str) -> Tuple[Optional[str], str]:
        """Parse 'Speaker: text' format. Returns (speaker, text)."""
        if ":" in dialogue:
            idx = dialogue.find(":")
            if idx > 0 and idx < len(dialogue) - 1:
                if dialogue[idx-1].isdigit() and dialogue[idx+1].isdigit():
                    idx = dialogue.find(":", idx + 1)
            
            if idx != -1:
                speaker_candidate = dialogue[:idx].strip()
                text_content = dialogue[idx+1:].strip()
                if (len(speaker_candidate) < 40 and 
                    "\n" not in speaker_candidate and
                    not any(char in speaker_candidate for char in ("?", "!"))):
                    return speaker_candidate, text_content
        return None, dialogue.strip()

    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """Word-wrap rich text, respecting explicit newlines and actual bold/italic font widths."""
        paragraphs = text.split('\n')
        all_wrapped_lines = []
        
        for p_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if p_idx > 0 and p_idx < len(paragraphs) - 1:
                    all_wrapped_lines.append("")
                continue
                
            # Step 1: Parse the paragraph into styled segments
            segments = self._parse_rich_text(paragraph)
            
            # Step 2: Build a list of characters with their associated style
            char_styles = []
            for val, style in segments:
                for char in val:
                    char_styles.append((char, style))
                    
            if not char_styles:
                continue
                
            # Step 3: Reconstruct into tokens (words and spaces)
            tokens = []
            current_token_chars = []
            current_style = char_styles[0][1]
            current_is_space = char_styles[0][0].isspace()
            
            for char, style in char_styles:
                is_space = char.isspace()
                if style != current_style or is_space != current_is_space:
                    if current_token_chars:
                        tokens.append(("".join(current_token_chars), current_style, current_is_space))
                    current_token_chars = [char]
                    current_style = style
                    current_is_space = is_space
                else:
                    current_token_chars.append(char)
            if current_token_chars:
                tokens.append(("".join(current_token_chars), current_style, current_is_space))
                
            # Step 4: Wrap tokens into lines
            wrapped_lines_tokens = []
            current_line_tokens = []
            current_line_width = 0
            size = font.size if hasattr(font, "size") else self.base_font_size
            
            for token_text, style, is_space in tokens:
                tfont = self._get_font(size, style)
                token_w = self._text_width(token_text, tfont)
                
                if current_line_width + token_w <= max_width:
                    current_line_tokens.append((token_text, style, is_space))
                    current_line_width += token_w
                else:
                    if is_space:
                        continue
                    if current_line_tokens:
                        while current_line_tokens and current_line_tokens[-1][2]:
                            current_line_tokens.pop()
                        if current_line_tokens:
                            wrapped_lines_tokens.append(current_line_tokens)
                    current_line_tokens = [(token_text, style, is_space)]
                    current_line_width = token_w
                    
            if current_line_tokens:
                while current_line_tokens and current_line_tokens[-1][2]:
                    current_line_tokens.pop()
                if current_line_tokens:
                    wrapped_lines_tokens.append(current_line_tokens)
                    
            # Step 5: Convert wrapped token lines back to strings with markdown tags
            for line_tokens in wrapped_lines_tokens:
                line_str = ""
                prev_style = "regular"
                
                for token_text, style, is_space in line_tokens:
                    if style == prev_style:
                        line_str += token_text
                    else:
                        if prev_style == "bold":
                            line_str += "**"
                        elif prev_style == "italic":
                            line_str += "*"
                            
                        if style == "bold":
                            line_str += "**" + token_text
                        elif style == "italic":
                            line_str += "*" + token_text
                        else:
                            line_str += token_text
                            
                        prev_style = style
                        
                if prev_style == "bold":
                    line_str += "**"
                elif prev_style == "italic":
                    line_str += "*"
                    
                all_wrapped_lines.append(line_str)
                
            if p_idx < len(paragraphs) - 1:
                all_wrapped_lines.append("")
                
        return all_wrapped_lines if all_wrapped_lines else [""]

    def _text_width(self, text: str, font) -> int:
        """Get the pixel width of a text string."""
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except Exception:
            return len(text) * 8

    def _get_font(self, size: int, variant: str = "regular") -> Any:
        """Get or create a font at the given size and variant (regular, bold, italic)."""
        cache_key = (size, variant.lower())
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = None
        
        # Determine local font paths for Comic Neue
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "fonts")
        local_comic_font = os.path.join(font_dir, f"ComicNeue-{variant.capitalize()}.ttf")
        
        # Candidates list for regular, bold, and italic variants
        font_candidates = {
            "regular": [
                local_comic_font,
                "arial.ttf", "Arial.ttf",
                "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "bold": [
                local_comic_font,
                "arialbd.ttf", "Arialbd.ttf", "Arial-Bold.ttf",
                "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ],
            "italic": [
                local_comic_font,
                "ariali.ttf", "Ariali.ttf", "Arial-Italic.ttf",
                "DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf",
                "C:/Windows/Fonts/ariali.ttf",
                "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
                "/Library/Fonts/Arial Italic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            ]
        }

        # Try user-defined font path first (only if style is regular, since single path provided)
        if variant.lower() == "regular" and self.font_path and os.path.exists(self.font_path):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception:
                pass

        # Try candidates based on variant
        if font is None:
            candidates = font_candidates.get(variant.lower(), font_candidates["regular"])
            for fp in candidates:
                try:
                    font = ImageFont.truetype(fp, size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font
