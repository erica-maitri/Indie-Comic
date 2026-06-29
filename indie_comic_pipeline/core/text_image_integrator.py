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
                 ollama_url: str = "http://localhost:11434",
                 dry_run: bool = False):
        self.font_path = font_path
        self.base_font_size = base_font_size
        self.max_bubble_width_ratio = max_bubble_width_ratio
        self.output_dir = output_dir
        self.ollama_model = ollama_model
        self.ollama_url = os.environ.get("OLLAMA_URL") or ollama_url
        self.dry_run = dry_run
        self._font_cache: Dict[int, Any] = {}
        self._llm = None
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

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

JSON structure:
{
  "speaker": "name of speaker or null",
  "dialogue_clean": "dialogue text without character name prefix",
  "bubble_shape": "ellipse|dashed_ellipse|jagged|cloud|spiky",
  "speaker_position": "left|center|right",
  "font_scale": 1.0,
  "x_ratio": 0.5,
  "y_ratio": 0.15
}"""

        prompt = f"""Panel ID: {panel_id}
Dialogue: "{dialogue}"
Emotion Beat: {emotion_beat}
Scene/Action Description: "{scene_desc or 'Not specified'}"

Please design the speech bubble layout. Determine the speaker, clean dialogue, bubble shape best suited to the emotional beat, relative horizontal position (left, right, or center) and the relative X (x_ratio) and Y (y_ratio) coordinates for bubble placement (ratios between 0.05 and 0.95). Keep the bubble y_ratio near the top (e.g. 0.15 to 0.3) so it doesn't cover main character bodies, but adjust slightly based on panel ID to avoid overlap.
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
                for key in ["speaker", "dialogue_clean", "bubble_shape", "speaker_position", "font_scale", "x_ratio", "y_ratio"]:
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

        if self.dry_run:
            result = image.copy().convert("RGBA")
            overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            w, h = result.size
            font_size = self.base_font_size
            font = self._get_font(font_size)
            
            speaker, text_clean = self._parse_dialogue(dialogue)
            if not text_clean:
                text_clean = dialogue
                
            display_text = f"{speaker}: {text_clean}" if speaker else text_clean
            wrapped_lines = self._wrap_text(display_text, font, w - 40)
            
            line_height = font_size + 4
            text_height = len(wrapped_lines) * line_height
            rect_h = text_height + 20
            
            # Semi-transparent rectangle overlay
            draw.rectangle([10, h - rect_h - 10, w - 10, h - 10], fill=(0, 0, 0, 180), outline=(255, 255, 255, 255), width=2)
            
            # Render lines
            y_pos = h - rect_h
            for line in wrapped_lines:
                draw.text((20, y_pos), line, fill=(255, 255, 255, 255), font=font)
                y_pos += line_height
                
            result = Image.alpha_composite(result, overlay)
            return result.convert("RGB")

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

        # Wrap text to fit bubble width
        wrapped_lines = self._wrap_text(text, font, max_bubble_w - 20)

        # Calculate text block size
        line_height = font_size + 4
        text_height = len(wrapped_lines) * line_height
        text_width = max(
            self._text_width(line, font) for line in wrapped_lines
        ) if wrapped_lines else 100

        # Bubble dimensions with padding
        padding = 14
        bubble_w = text_width + padding * 2
        bubble_h = text_height + padding * 2

        # If speaker name, add extra height
        if speaker:
            bubble_h += line_height

        # Position the bubble using ratios
        bx, by = self._calculate_bubble_position(
            w, h, bubble_w, bubble_h, speaker_pos, panel_id, x_ratio, y_ratio
        )

        # Draw the bubble shape
        self._draw_bubble(draw, bx, by, bubble_w, bubble_h, style)

        # Draw tail
        tail_x = self._get_tail_x(bx, bubble_w, speaker_pos, x_ratio)
        tail_y = by + bubble_h
        self._draw_tail(draw, tail_x, tail_y, style)

        # Draw speaker name (bold) if present
        text_y = by + padding
        if speaker:
            speaker_font = self._get_font(int(font_size * 0.85))
            draw.text((bx + padding, text_y), speaker.upper(),
                      fill=(60, 60, 60, 255), font=speaker_font)
            text_y += line_height

        # Draw dialogue text
        for line in wrapped_lines:
            draw.text((bx + padding, text_y), line,
                      fill=(20, 20, 20, 255), font=font)
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

        if shape in ("ellipse", "dashed_ellipse"):
            radius = min(w, h) // 4
            self._draw_rounded_rect(draw, x, y, w, h, radius, fill, border, border_w)

        elif shape == "jagged":
            radius = 6
            self._draw_rounded_rect(draw, x, y, w, h, radius, fill, border, border_w)

        elif shape == "cloud":
            radius = min(w, h) // 3
            self._draw_rounded_rect(draw, x, y, w, h, radius, fill, border, border_w)

        elif shape == "spiky":
            # Rectangular with sharp corners for shouts
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=border, width=border_w)

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

    def _draw_tail(self, draw: Any,
                   tip_x: int, tip_y: int,
                   style: Dict[str, Any]):
        """Draw the speech bubble tail pointing toward the speaker."""
        tail_style = style.get("tail_style", "smooth")
        fill = style.get("fill", (255, 255, 255, 230))
        border = style.get("border_color", (40, 40, 40))

        tail_height = 15
        tail_width = 12

        if tail_style == "bubbles":
            for i in range(3):
                r = 4 - i
                cy = tip_y + (i + 1) * 8
                draw.ellipse(
                    [tip_x - r, cy - r, tip_x + r, cy + r],
                    fill=fill, outline=border, width=1,
                )
        else:
            points = [
                (tip_x - tail_width // 2, tip_y),
                (tip_x + tail_width // 2, tip_y),
                (tip_x, tip_y + tail_height),
            ]
            draw.polygon(points, fill=fill, outline=border)

    # ─────────────────────────────────────────────────────────────────────
    # Positioning
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_bubble_position(self, img_w: int, img_h: int,
                                   bubble_w: int, bubble_h: int,
                                   speaker_pos: str,
                                   panel_id: int,
                                   x_ratio: Optional[float] = None,
                                   y_ratio: Optional[float] = None) -> Tuple[int, int]:
        """Calculate bubble position based on speaker position, panel ID, and optional planned ratios."""
        margin = 15

        if x_ratio is not None and y_ratio is not None:
            # Centered on ratio coordinates
            x = int(x_ratio * img_w - bubble_w // 2)
            y = int(y_ratio * img_h - bubble_h // 2)
        else:
            if speaker_pos == "left":
                x = margin
            elif speaker_pos == "right":
                x = img_w - bubble_w - margin
            else:
                x = (img_w - bubble_w) // 2

            y_offset = (panel_id % 3) * 60
            y = margin + y_offset

        # Clamp to image bounds
        x = max(margin, min(x, img_w - bubble_w - margin))
        y = max(margin, min(y, img_h - bubble_h - 40))

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
            # Avoid splitting on colons inside time stamps (e.g., 12:30)
            idx = dialogue.find(":")
            if idx > 0 and idx < len(dialogue) - 1:
                if dialogue[idx-1].isdigit() and dialogue[idx+1].isdigit():
                    # Find next colon
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
        """Word-wrap text to fit within max_width pixels."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if self._text_width(test_line, font) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text[:30]]

    def _text_width(self, text: str, font) -> int:
        """Get the pixel width of a text string."""
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except Exception:
            return len(text) * 8

    def _get_font(self, size: int) -> Any:
        """Get or create a font at the given size."""
        if size in self._font_cache:
            return self._font_cache[size]

        font = None
        if self.font_path and os.path.exists(self.font_path):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception:
                pass

        if font is None:
            font_candidates = [
                "arial.ttf", "Arial.ttf",
                "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for fp in font_candidates:
                try:
                    font = ImageFont.truetype(fp, size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font
