import os
import glob

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

data_dir = "./data/text"
data_paths = glob.glob(os.path.join(data_dir, "**"), recursive=True)
data_files = [item_path for item_path in data_paths if os.path.isfile(item_path)]

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[START]", "[END]"])

tokenizer.train(data_files, trainer)
tokenizer.save("hi_bpe_tokenizer.json")
# tokenizer = Tokenizer.from_file("hi_bpe_tokenizer.json")

output = tokenizer.encode("नमस्ते, आप कैसे हैं 😁?")
print(output.tokens) # ['नम', 'स्ते', ', ', 'आप', ' कैसे ', 'हैं ', '?']
