import torch
import sentencepiece as spm

def train_bpe_tokenizer(data_path, model_prefix='bpe', vocab_size=500):
    spm.SentencePieceTrainer.train(
        input=data_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        max_sentence_length=16384,
        split_by_unicode_script=1,
        split_by_number=1,
        split_by_whitespace=1,
        split_digits=0,
        treat_whitespace_as_suffix=0,
        allow_whitespace_only_pieces=0,
        byte_fallback=0,
        vocabulary_output_piece_score=1,
        hard_vocab_limit=1,
        use_all_vocab=0
    )

def load_tokenizer(model_prefix='bpe'):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp
