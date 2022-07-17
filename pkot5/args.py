from transformers import IntervalStrategy

default_args = dict(
    seed=42,
    data_seed=2205141535,
    learning_rate=1e-2,
    max_steps=1_000_000,
    per_device_train_batch_size=2,  # total_batch_size = 8 nodes * per_device_train_batch_size * gradient_accumulation_steps  => 256
    gradient_accumulation_steps=16,

    evaluation_strategy=IntervalStrategy.NO,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=100000,
)

xl_args = dict(
    seed=42,
    data_seed=2205141535,
    learning_rate=1e-2,
    max_steps=1_000_000,
    per_device_train_batch_size=4,  # total_batch_size = 2 nodes * per_device_train_batch_size * gradient_accumulation_steps  => 128
    gradient_accumulation_steps=16,

    evaluation_strategy=IntervalStrategy.NO,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=100000,

    pipeline_parallelism=2,
)

small_config = {
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 1024,
    "d_kv": 64,
    "d_model": 512,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 8,
    "num_heads": 6,
    "num_layers": 8,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "vocab_size": 50358
}

base_config = {
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 2048,
    "d_kv": 64,
    "d_model": 768,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 12,
    "num_heads": 12,
    "num_layers": 12,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "vocab_size": 50358
}

large_config = {
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 2816,
    "d_kv": 64,
    "d_model": 1024,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 24,
    "num_heads": 16,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "vocab_size": 50358,
}

xl_config = {
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 5120,
    "d_kv": 64,
    "d_model": 2048,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 24,
    "num_heads": 32,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "vocab_size": 50358
}

ARGS = {
    "small": default_args,
    "base": default_args,
    "large": default_args,
    "xl": xl_args,
}

CONFIGS = {
    "small": small_config,
    "base": base_config,
    "large": large_config,
    "xl": xl_config,
}
