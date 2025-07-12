import re
import nltk
import logging
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from typing import List, Union

logger = logging.getLogger(__name__)

_nltk_loaded = False

def _load_nltk_resources():
    global _nltk_loaded
    if _nltk_loaded:
        return
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        _nltk_loaded = True
        logger.info("NLTK resources loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load NLTK resources: {e}")
        logger.info("Will use simple processing without NLTK")
        _nltk_loaded = True

_load_nltk_resources()

try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize NLTK: {e}")
    logger.info("Will use simple processing")
    stop_words = set()
    lemmatizer = None

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(
    text: str,
    return_tokens: bool = False,
    min_length: int = 2,
    remove_stopwords: bool = True,
    use_lemmatization: bool = True
) -> Union[str, List[str]]:
    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""
    try:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        if _nltk_loaded and stop_words and lemmatizer:
            tokens = nltk.word_tokenize(text)
            pos_tags = pos_tag(tokens)
            filtered_tokens = []
            for token, tag in pos_tags:
                if len(token) < min_length:
                    continue
                if remove_stopwords and token in stop_words:
                    continue
                if use_lemmatization and lemmatizer:
                    token = lemmatizer.lemmatize(token, get_wordnet_pos(tag))
                if token:
                    filtered_tokens.append(token)
        else:
            tokens = text.split()
            filtered_tokens = [token for token in tokens if len(token) >= min_length]
        if return_tokens:
            return filtered_tokens
        else:
            return ' '.join(filtered_tokens)
    except Exception as e:
        logger.error(f"Error in text processing: {e}")
        try:
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            tokens = text.split()
            filtered_tokens = [token for token in tokens if len(token) >= min_length]
            if return_tokens:
                return filtered_tokens
            else:
                return ' '.join(filtered_tokens)
        except:
            return [] if return_tokens else ""

def preprocess_for_indexing(text: str, min_length: int = 2) -> List[str]:
    result = preprocess(text, return_tokens=True, min_length=min_length)
    if isinstance(result, list):
        return result
    elif isinstance(result, str):
        return result.split() if result else []
    return []

def preprocess_for_vectorization(text: str, min_length: int = 2) -> str:
    result = preprocess(text, return_tokens=False, min_length=min_length)
    if isinstance(result, str):
        return result
    elif isinstance(result, list):
        return ' '.join(result)
    return ""

def get_preprocessing_stats(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            'original_length': 0,
            'processed_length': 0,
            'tokens_count': 0,
            'stopwords_removed': 0,
            'short_words_removed': 0
        }
    try:
        original_length = len(text)
        text_lower = text.lower()
        text_clean = re.sub(r'[^a-z0-9\s]', '', text_lower)
        if _nltk_loaded and stop_words:
            all_tokens = nltk.word_tokenize(text_clean)
            stopwords_count = sum(1 for token in all_tokens if token in stop_words)
        else:
            all_tokens = text_clean.split()
            stopwords_count = 0
        short_words_count = sum(1 for token in all_tokens if len(token) < 2)
        processed_tokens = preprocess_for_indexing(text)
        return {
            'original_length': original_length,
            'processed_length': len(' '.join(processed_tokens)),
            'tokens_count': len(processed_tokens),
            'stopwords_removed': stopwords_count,
            'short_words_removed': short_words_count
        }
    except Exception as e:
        logger.error(f"Error calculating preprocessing statistics: {e}")
        return {
            'original_length': len(text) if text else 0,
            'processed_length': 0,
            'tokens_count': 0,
            'stopwords_removed': 0,
            'short_words_removed': 0
        }
