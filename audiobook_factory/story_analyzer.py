import os
import shutil
import csv
from booknlp.booknlp import BookNLP

class StoryAnalyzer:
    def __init__(self, output_dir, model_size="small", enable=False):
        self.output_dir = output_dir
        self.enable = enable
        
        if self.enable:
            print(f"Loading BookNLP ({model_size}) model...")
            model_params = {
                "pipeline": "entity,quote,supersense,event,coref", 
                "model": model_size, 
            }
            try:
                self.booknlp = BookNLP("en", model_params)
            except Exception as e:
                print(f"Failed to load BookNLP: {e}")
                print("Try running: python -m spacy download en_core_web_sm")
                self.enable = False

    def analyze_text(self, text, chapter_id):
        """
        Runs BookNLP on the text and returns a list of tagged segments.
        Structure: [{"text": "...", "speaker": "Narrator" | "CharacterName"}]
        """
        if not self.enable:
            # Pass-through if disabled
            return [{"text": text, "speaker": "Narrator"}]

        # BookNLP works on files. Write temp file.
        temp_input = os.path.join(self.output_dir, f"temp_{chapter_id}.txt")
        temp_output_dir = os.path.join(self.output_dir, f"booknlp_{chapter_id}")
        
        with open(temp_input, "w", encoding="utf-8") as f:
            f.write(text)
            
        # Run Analysis
        print(f"Running BookNLP analysis for {chapter_id}...")
        self.booknlp.process(temp_input, temp_output_dir, f"chapter_{chapter_id}")
        
        # Parse Results
        segments = self._parse_booknlp_output(temp_output_dir, f"chapter_{chapter_id}")
        
        # Cleanup
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
            
        return segments

    def _parse_booknlp_output(self, output_dir, file_id):
        """
        Reads .tokens and .quotes files to reconstruct the text with speaker tags.
        """
        tokens_file = os.path.join(output_dir, f"{file_id}.tokens")
        
        if not os.path.exists(tokens_file):
            return []

        # 1. Load Tokens (we need the words and their offsets)
        tokens = []
        with open(tokens_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                tokens.append({
                    "word": row["word"],
                    "token_id": int(row["tokenId"]),
                    "speaker": "Narrator" # Default
                })

        # 2. Load Quotes and map to tokens
        quotes_file = os.path.join(output_dir, f"{file_id}.quotes")
        if os.path.exists(quotes_file):
            with open(quotes_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    start_token = int(row["quoteStart"])
                    end_token = int(row["quoteEnd"])
                    speaker = row["mention"]  # The resolved character name
                    
                    # Tag the tokens
                    # Note: Handle bounds safely
                    for i in range(start_token, min(end_token + 1, len(tokens))):
                        tokens[i]["speaker"] = speaker

        # 3. Reconstruct Segments (Merge consecutive tokens with same speaker)
        segments = []
        if not tokens: return []
        
        current_segment = {"text": tokens[0]["word"], "speaker": tokens[0]["speaker"]}
        
        # Helper to handle spacing roughly (BookNLP provides byte offsets but simple join is often used)
        # For TTS, we want natural spacing.
        # We'll assume space between tokens unless punctuation says otherwise?
        # Actually BookNLP tokens usually include punctuation as separate tokens.
        
        for i in range(1, len(tokens)):
            token = tokens[i]
            word = token["word"]
            speaker = token["speaker"]
            
            # Simple heuristic for spacing:
            # If it's punctuation, attach to previous. Else add space.
            # This is naive. Better: Check offset difference if available, but for now:
            is_punct = not word.isalnum()
            prefix = "" if is_punct else " " 
            
            if speaker == current_segment["speaker"]:
                current_segment["text"] += prefix + word
            else:
                segments.append(current_segment)
                current_segment = {"text": word, "speaker": speaker}
                
        segments.append(current_segment)
        
        return segments
