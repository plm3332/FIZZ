import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

model_map = {
    "tals": {"model_ckpt": "tals/albert-xlarge-vitaminc-mnli"}
}

class AtomicFactFilterer:
    def __init__(self, model_name="tals", device="cuda"):
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)

        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt,
                                                       use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt)
        self.model.to(self.device)

    # from https://github.com/tingofurro/summac/
    def split_sentences(self, text):
        sentences = sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        return sentences
    
    def atomic_facts_filtering(self, original_text, decomposed_text):
        if self.model == None:
            self.load_lm()
        original_sentences_list = self.split_sentences(original_text)
        decomposed_sentences_list = self.split_sentences(decomposed_text)
        filtered_atomic_facts = ""
        for decomposed_sentence in decomposed_sentences_list:
            is_ok = False
            for original_sentence in original_sentences_list:
                features = self.tokenizer([original_sentence.strip()],
                                          [decomposed_sentence.strip()],
                                          padding=True,
                                          truncation=True,
                                          return_tensors="pt").to(self.device)
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(**features).logits
                    scores = torch.nn.functional.softmax(logits, dim=-1)
                    evid_score = np.array(scores[0][0].cpu()).item()
                    conts_score = np.array(scores[0][1].cpu()).item()
                    neuts_score = np.array(scores[0][2].cpu()).item()
                    if evid_score > conts_score and evid_score > neuts_score:
                        is_ok = True
                        break
            if is_ok:
                filtered_atomic_facts += decomposed_sentence.strip() + " "
            else:
                continue
        if filtered_atomic_facts == "":
            return original_text
        else:
            return filtered_atomic_facts.strip()

def main():
    filterer = AtomicFactFilterer()
    original = "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."
    atomic_facts = "Lisa Courtney is from Hertfordshire. Lisa Courtney has spent most of her life collecting Pokémon memorabilia."
    filtered_atomic_facts = filterer.atomic_facts_filtering(original, atomic_facts)
    print(filtered_atomic_facts)

if __name__ == "__main__":
    main()