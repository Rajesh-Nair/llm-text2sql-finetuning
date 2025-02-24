import os
from dataclasses import dataclass, field
from typing import Optional
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def create_datasets(tokenizer, data_args, apply_chat_template=False):

    def preprocess(samples):
        batch = []
        for sample in zip(samples["context"],samples["question"],samples["answer"]): 
            conversation = [{"role":"system", "content":sample[0]}, {"role":"user", "content":sample[1]}, {"role":"assistant", "content":sample[2]}]
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}
    
    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split.strip())
            raw_datasets[split.strip()] = dataset
        except DatasetGenerationError:
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split.strip()))
            raw_datasets[split.strip()] = dataset
        except ValueError :
            print("Dataset {} data not available in the source".format(split))

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
    train_data = raw_datasets["train"]
    valid_data = raw_datasets.get("test", None)
    if valid_data is None:
        print("Spliting train data 90:10 since no test data available")
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        train_data = raw_datasets["train"]
        valid_data = raw_datasets["test"]

    return train_data, valid_data


if __name__ == "__main__":
    # For testing
    from dataclasses import dataclass, field
    from transformers import AutoTokenizer
    from datasets.arrow_dataset import Dataset as dataset_class

    @dataclass
    class data_args :
        dataset_name : str = field(default="b-mc2/sql-create-context")
        splits : str = field(default="train,test")


    @dataclass
    class model_args :
        model_name_or_path : str = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # Load dataset
    train_data, valid_data = create_datasets(tokenizer, data_args, apply_chat_template=True)

    # Test cases
    assert(train_data) == dataset_class
    assert(valid_data) == dataset_class