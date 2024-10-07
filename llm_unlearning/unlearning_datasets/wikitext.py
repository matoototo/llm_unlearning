import datasets
from llm_unlearning.unlearning_datasets.rawtext import RawTextDataset

class WikiTextDataset(RawTextDataset):
    def _load_dataset(self):
        dataset = datasets.load_dataset("wikitext", "wikitext-103-v1", split="train")
        if self.full_context_mode:
            return self._create_full_context_items(dataset)
        return dataset
