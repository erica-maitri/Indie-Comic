import os
import json
import time
from collections import defaultdict

class IncrementalLearner:
    """
    Collects user feedback on generated comic panels and extracts patterns
    to improve prompt generation over time.
    """
    
    def __init__(self, db_path="outputs/feedback_db.json"):
        self.db_path = db_path
        self.feedback_db = []
        self.prompt_modifiers = defaultdict(list)
        self._load_db()
        
    def _load_db(self):
        """Loads existing feedback database if available"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.feedback_db = data.get('feedback', [])
                    self.prompt_modifiers = defaultdict(list, data.get('modifiers', {}))
            except json.JSONDecodeError:
                print("[!] Could not parse feedback DB. Starting fresh.")
                
    def _save_db(self):
        """Saves current state to database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump({
                'feedback': self.feedback_db,
                'modifiers': dict(self.prompt_modifiers)
            }, f, indent=4)
            
    def learn_from_feedback(self, panel_id: str, prompt: str, rating: int, comment: str):
        """
        Ingests user feedback for a specific panel.
        Rating should be 1-5.
        """
        feedback_entry = {
            'panel_id': panel_id,
            'prompt': prompt,
            'rating': rating,
            'comment': comment,
            'timestamp': time.time()
        }
        
        self.feedback_db.append(feedback_entry)
        
        # Analyze patterns if we have enough data
        if len(self.feedback_db) % 5 == 0:
            self._update_prompt_templates()
            
        self._save_db()
        print(f"[*] Feedback logged. Total entries: {len(self.feedback_db)}")
        
    def _update_prompt_templates(self):
        """
        Analyzes high vs low rated panels to adjust generation strategy.
        In a full ML implementation, this could fine-tune a LoRA.
        Here we extract explicit keywords from high-rated comments to build preferred modifiers.
        """
        high_rated = [f for f in self.feedback_db if f['rating'] >= 4]
        
        # Super simple NLP: if user comments "more contrast", we might append that
        # to future prompts.
        for item in high_rated:
            comment = item.get('comment', '').lower()
            if 'love the shadows' in comment or 'dark' in comment:
                if 'dark shadows' not in self.prompt_modifiers['noir']:
                    self.prompt_modifiers['noir'].append('dark shadows')
                    
    def get_enhanced_prompt(self, base_prompt: str, style: str) -> str:
        """Applies learned modifiers to a prompt"""
        modifiers = self.prompt_modifiers.get(style, [])
        if modifiers:
            mod_str = ", ".join(modifiers)
            return f"{base_prompt}, {mod_str}"
        return base_prompt
