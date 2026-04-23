import os
import re
import glob
import requests
from collections import Counter
import pickle

def download_raw_text(text_list):
    os.makedirs('raw_text', exist_ok=True)

    with open(text_list, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            data = line.split(',')
            id = data[0].strip()
            title = data[1].strip()
            author = data[2].strip()
            url = f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
            print(f"Attempting to download {title} by {author}")
            safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
            try:
                response = requests.get(url)
                response.raise_for_status()

                filename = os.path.join('raw_text', f"{safe_title}.txt")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            except Exception as e:
                print(e)
                try:
                    url = f"https://www.gutenberg.org/files/{id}/{id}.txt"
                    response = requests.get(url)
                    response.raise_for_status()

                    filename = os.path.join('raw_text', f"{safe_title}.txt")
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                except Exception as e:
                    print(f"Failed to download {title} by {author} from {url}:\n{e}")
                    print("Work not accessible")

def clean_text(text):
	# Gutenberg header/footer can be a bit inconsistent, so we check both markers
	start_markers = ["*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"]
	end_markers = ["*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"]

	start_pos = -1
	for marker in start_markers:
		start_pos = text.find(marker)
		if start_pos != -1:
			break
	if start_pos == -1:
		print(f"Failed on {text[:300]}")

	end_pos = -1
	for marker in end_markers:
		end_pos = text.find(marker)
		if end_pos != -1:
			break
	if end_pos == -1:
		print(f"Failed on {text[:300]}")

	# from start to end of text
	text = text[start_pos:end_pos].strip()

	# Normalize quotes and dashes so that they are the same throughout
	text = re.sub(r'[""''‚„]', '"', text)  # Normalize quotes
	text = re.sub(r'[–—−]', '-', text)     # Normalize dashes
	text = re.sub(r'[…]', '...', text)     # Normalize ellipsis

	# Remove non-printable characters except common whitespace
	# Sometimes there are weird unicode characters in the text
	text = ''.join(char for char in text if char.isprintable() or char in '\n\t ')

	# Remove illustration markers
	text = re.sub(r'\[Illustration[^\]]*\]', '', text, flags=re.IGNORECASE)

	# Remove editor's notes
	text = re.sub(r'\[Editor\'?s? [Nn]ote[^\]]*\]', '', text)

	# Remove page markers
	text = re.sub(r'\[Page \d+\]', '', text)

	# Remove transcriber notes
	text = re.sub(r'\[Transcriber\'?s? [Nn]ote[^\]]*\]', '', text)

	# Remove proofreader notes
	text = re.sub(r'\[Proofreader[^\]]*\]', '', text)
	# Remove footnote markers
	text = re.sub(r'\[Footnote[^\]]*\]', '', text)
	text = re.sub(r'\[\d+\]', '', text)

	# Remove parenthetical page references
	text = re.sub(r'\(p\. \d+\)', '', text)
	text = re.sub(r'\(pp\. \d+-\d+\)', '', text)
	# Remove chapter headings (various formats)
	text = re.sub(r'^CHAPTER [IVXLCDM\d]+\.?\s*$', '', text, flags=re.MULTILINE)
	text = re.sub(r'^Chapter \d+\.?\s*$', '', text, flags=re.MULTILINE)
	text = re.sub(r'^\d+\.?\s*$', '', text, flags=re.MULTILINE)

	# Remove section dividers
	text = re.sub(r'^[*\-_=]{3,}\s*$', '', text, flags=re.MULTILINE)
	# Remove excessive spacing around punctuation
	text = re.sub(r'\s+([,.!?;:])', r'\1', text)
	text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)

	# Remove excessive blank lines (more than 2 consecutive)
	text = re.sub(r'\n{3,}', '\n\n', text)

	# Remove trailing whitespace from lines
	text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

	# Remove leading/trailing whitespace from entire text
	text = text.strip()

	return text

def process_all_texts(folder):
	os.makedirs('processed_text', exist_ok=True)
	combined = ''
	for file in glob.glob(os.path.join(folder, '*')):
		with open(file.strip(), 'r', encoding='utf-8') as f:
			text = f.read()
			text = clean_text(text)
			combined += text + '\n\n\n\n'
		filename = os.path.basename(file)
		new_file = os.path.join('processed_text', filename)
		with open(new_file, 'w', encoding='utf-8') as f:
			f.write(text)
	with open('combined_mystery.txt', 'w', encoding='utf-8') as f:
		f.write(combined)

def tokenize_data(combined_text_file, vocab_size=10000):
	"""Simple word-level tokenization with vocabulary limit

	Args:
		combined_text_file: Path to combined cleaned text file (contains all text in dataset)
		vocab_size: Maximum vocabulary size (including special tokens)
	Returns:
		tokens: List of token ids for the entire text
		word_to_idx: Dictionary mapping words to their token ids
		vocab: List of words in the vocabulary (index corresponds to token id)
	"""
	with open(combined_text_file, 'r', encoding='utf-8') as f:
		text = f.read()

	# Basic tokenization (split on whitespace and punctuation)
	words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())

	# Count frequencies and take top words
	word_counts = Counter(words)
	most_common = word_counts.most_common(vocab_size - 3)  # Reserve 3 special tokens

	# Create vocabulary
	vocab = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, _ in most_common]
	word_to_idx = {word: i for i, word in enumerate(vocab)}

	# Tokenize with UNK for rare words
	tokens = []
	for word in words:
		tokens.append(word_to_idx.get(word, 1))  # idx 1 is <UNK>

	return tokens, word_to_idx, vocab

def train_test_split(data, train_fraction):
	train_data = data[:int(train_fraction * len(data))]
	test_data = data[int(train_fraction * len(data)):]
	return train_data, test_data

def full_pipeline():
	download_raw_text('text_list.csv')
	print("All texts downloaded successfully!")
	process_all_texts('raw_text')
	print("All texts processed successfully!")
	tokens, word_to_idx, vocab = tokenize_data('combined_mystery.txt')

	train_data, test_data = train_test_split(tokens, 0.8)
	print("Data tokenized and split successfully!")

	data_dict = {'train_data': train_data, 'test_data': test_data, 'vocab': vocab, 'word_to_idx': word_to_idx}
	with open('mystery_data.pkl', 'wb') as f:
		pickle.dump(data_dict, f)
	print("Data processing complete and saved to mystery_data.pkl.")

if __name__ == '__main__':
	full_pipeline()
