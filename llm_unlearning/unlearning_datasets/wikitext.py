import datasets
from llm_unlearning.unlearning_datasets.rawtext import RawTextDataset

class WikiTextDataset(RawTextDataset):
    def _load_dataset(self):
        dataset = datasets.load_dataset("wikitext", "wikitext-103-v1", split="train")
        text = ""
        size = 0
        max_size = self.config.get("max_size_mb", 100) * 1024 * 1024
        for item in dataset:
            if not item["text"].strip(): continue
            item_size = len(item["text"].encode('utf-8'))
            if size + item_size > max_size: break
            text += item["text"] + " "
            size += item_size

        return self.tokenizer.encode(text.strip(), add_special_tokens=False)
