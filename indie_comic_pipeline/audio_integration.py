import os
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
import tempfile
import uuid

class AudioIntegrator:
    """Integrates Text-to-Speech (TTS) capabilities into the comic generator"""
    
    def __init__(self, output_dir="outputs/audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # We can map characters to different TLDs to simulate different accents
        self.voice_profiles = {
            'Spider-Man': {'tld': 'com', 'lang': 'en'}, # Default American
            'Batman': {'tld': 'co.uk', 'lang': 'en'},   # British accent for flair
            'Wolverine': {'tld': 'ca', 'lang': 'en'},   # Canadian accent
            'Default': {'tld': 'com', 'lang': 'en'}
        }
    
    def get_character_voice(self, character: str) -> dict:
        """Retrieve the voice settings for a given character"""
        return self.voice_profiles.get(character, self.voice_profiles['Default'])
    
    def generate_audio_dialogue(self, dialogue: str, character: str) -> str:
        """
        Generate audio for a given dialogue and character
        Returns the path to the saved MP3 file
        """
        if not dialogue or dialogue.strip() == "...":
            return None
        
        if not GTTS_AVAILABLE:
            print("[!] gTTS not installed. Run: pip install gTTS")
            return None
            
        voice = self.get_character_voice(character)
        
        try:
            # Generate the TTS
            tts = gTTS(text=dialogue, lang=voice['lang'], tld=voice['tld'], slow=False)
            
            # Save to file
            filename = f"dialogue_{uuid.uuid4().hex[:8]}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            tts.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"[!] TTS Generation failed: {e}")
            return None

    def align_with_panel(self, audio_path: str, dialogue: str) -> dict:
        """Returns alignment metadata for interactive playback"""
        # Simple alignment structure
        return {
            'audio_file': audio_path,
            'text': dialogue,
            'duration_estimate': len(dialogue.split()) * 0.4 # rough estimate: 0.4s per word
        }
    
    def create_interactive_comic(self, pages: list, audio_tracks: dict, output_path: str = "outputs/interactive_comic.html"):
        """
        Creates an interactive HTML file where clicking panels plays their audio
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive AI Comic</title>
            <style>
                body { background-color: #1a1a1a; color: white; display: flex; flex-direction: column; align-items: center; }
                .panel { margin: 20px; cursor: pointer; border: 2px solid transparent; transition: border-color 0.3s; }
                .panel:hover { border-color: #bb86fc; }
            </style>
        </head>
        <body>
            <h1>Interactive AI Comic</h1>
            <p>Click on the panels to hear the dialogue!</p>
            <div id="comic-container">
        """
        
        # Inject panels and scripts
        # This is a simplified mockup of how the data would be structured in the UI
        # In production, we'd base64 encode or link the images/audio.
        
        html_content += """
            </div>
            <script>
                function playAudio(src) {
                    let audio = new Audio(src);
                    audio.play();
                }
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path
