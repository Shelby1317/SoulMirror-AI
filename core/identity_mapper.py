from transformers import pipeline

class IdentityMapper:
    def __init__(self):
        self.analyzer = pipeline("text-generation", model="gpt2")
    
    def analyze_journal_entry(self, text, user_values=None):
        prompt = f"""
        You are SoulMirror AI. Reflect the user's inner identity terrain.
        Do NOT give advice. Mirror patterns, tensions, latent potentials.

        User Values (if any): {user_values or 'Not specified'}

        Journal Entry:
        "{text}"

        Reflection Guidelines:
        - Identify core values expressed or contradicted
        - Note emotional undercurrents
        - Surface hidden contradictions or harmonies
        - Use poetic, gentle, non-judgmental language
        - End with an open question that invites deeper self-dialogue

        Reflection:
        """
        
        result = self.analyzer(prompt, max_new_tokens=200, temperature=0.7)
        return result[0]['generated_text'].strip()
