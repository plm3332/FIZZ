from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import prompts_example
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import sent_tokenize

model_map = {
    "orca2": {"model_ckpt": "microsoft/Orca-2-7b"},
    "mistral": {"model_ckpt": "mistralai/Mistral-7B-Instruct-v0.2"},
    "zephyr": {"model_ckpt": "HuggingFaceH4/zephyr-7b-beta"}
}

class AtomicFactDecomposer:
    def __init__(self, model_name="orca2", device="cuda"):
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)
        
        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt,
                                                       use_fast=False,
                                                       padding_side="left",
                                                       add_special_tokens=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_ckpt,
                                                          torch_dtype=torch.float16)
        self.model.to(self.device)

    # from https://github.com/tingofurro/summac/
    def split_sentences(self, text):
        sentences = sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        return sentences
    
    def get_prompt(self, text):
        if self.model_name == "orca2":
            return prompts_example.orca_format(text)
        elif self.model_name == "mistral":
            return prompts_example.mistral_format(text)
        elif self.model_name == "zephyr":
            return prompts_example.zephyr_format(text)
    
    def tokenize(self, text):
        if self.model_name =="orca2":
            model_inputs = self.tokenizer(self.get_prompt(text), 
                                          return_tensors="pt")
            return model_inputs['input_ids']
        else:
            model_inputs = self.tokenizer.apply_chat_template(self.get_prompt(text),
                                                              return_tensors="pt")
            return model_inputs
    
    def generate_not_batch(self, input_id):
        with torch.no_grad():
            generated_id = self.model.generate(torch.tensor(input_id).to(self.device),
                                               max_new_tokens=256,
                                               do_sample=False,
                                               pad_token_id=self.tokenizer.eos_token_id)
        return generated_id
    
    def decode_not_batch(self, generated_id):
        decoded_id = self.tokenizer.batch_decode(generated_id, skip_special_tokens=True)
        return decoded_id
    
    def generate_and_decode_not_batch(self, tokenized_id):
        generated_id = self.generate_not_batch(tokenized_id)
        decoded_id = self.decode_not_batch(generated_id)
        return decoded_id

    def preprocess_decoded_ids(self, decoded_id):
        if self.model_name == 'orca2':
            text = decoded_id[0].split('<|im_start|> assistant')[-1]
            preprocessed_text = "".join(text.split('\n-')).replace('\n', '')
            preprocessed_text = preprocessed_text.replace("<|im_start|>", "")
            preprocessed_text = preprocessed_text.replace("Some possible atomic facts of the text are:", "")
        elif self.model_name == 'zephyr':
            text = decoded_id[0].split('<|assistant|>')[-1]
            preprocessed_text = "".join(text.split('\n-')).replace('\n','')
        elif self.model_name == 'mistral':
            text = decoded_id[0].split('[/INST]')[-1]
            preprocessed_text = "".join(text.split('\n-')).replace('\n', '')
        return preprocessed_text
    
    def atomic_facts_decompose(self, text):
        if self.model == None:
            self.load_lm()
        sentences_list = self.split_sentences(text)
        full_atomic_facts = "" 
        for sentence in sentences_list:
            input_ids = self.tokenize(sentence)
            decoded_ids = self.generate_and_decode_not_batch(input_ids)
            preprocessed_ids = self.preprocess_decoded_ids(decoded_ids)
            full_atomic_facts += preprocessed_ids
        return full_atomic_facts
    
def main():
    decomposer = AtomicFactDecomposer()
    text = "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."
    atomic_facts = decomposer.atomic_facts_decompose(text)
    print(atomic_facts)

if __name__ == "__main__":
    main()