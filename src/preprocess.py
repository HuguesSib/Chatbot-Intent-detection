import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

class Preprocess:
    def __init__(self, text_list):
        self.text_list = text_list

    def lowercase(self):
        lowercase_text = []
        for text in self.text_list:
            lowercase_text.append(text.lower())
        return lowercase_text

    def remove_special_chars(self):
        cleaned_text = []
        for text in self.text_list:
            cleaned_text.append(text.translate(str.maketrans("", "", string.punctuation)))
        return cleaned_text

    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))
        filtered_text = []
        for text in self.text_list:
            filtered_text.append(' '.join([word for word in text.split() if word.lower() not in stop_words]))
        return filtered_text

    def tokenize_words(self):
        tokenized_words = []
        for text in self.text_list:
            tokenized_words.append(word_tokenize(text))
        return tokenized_words

    def tokenize_sentences(self):
        tokenized_sentences = []
        for text in self.text_list:
            tokenized_sentences.append(sent_tokenize(text))
        return tokenized_sentences
