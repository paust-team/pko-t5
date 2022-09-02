from transformers import IntervalStrategy

DEFAULT_CONFIG = dict(
    seed=42,
    learning_rate=5e-5,
    optim="adafactor",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,

    generation_max_length=64,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.NO,
)

DEFAULT_ADAMW_CONFIG = dict(
    seed=42,
    learning_rate=3e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_ratio=0.06,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,

    generation_max_length=64,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.NO,
)


NLI_STS_CONFIG = dict(
    seed=42,
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=100,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,

    generation_max_length=64,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.EPOCH,
    # eval_steps=10,
    save_strategy=IntervalStrategy.NO,
)

MRC_CONFIG = dict(
    seed=42,
    learning_rate=3e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_ratio=0.06,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,

    generation_max_length=64,
    generation_num_beams=6,

    evaluation_strategy=IntervalStrategy.EPOCH,
    # eval_steps=10,
    save_strategy=IntervalStrategy.NO,
)

MRC_ALL_CONTEXT_CONFIG = dict(
    seed=42,
    learning_rate=1e-4,
    optim="adafactor",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,

    generation_max_length=64,
    generation_num_beams=2,

    evaluation_strategy=IntervalStrategy.EPOCH,
    # eval_steps=10,
    save_strategy=IntervalStrategy.NO,
)

DP_CONFIG = dict(
    seed=42,
    learning_rate=5e-5,
    optim="adafactor",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,

    generation_max_length=350,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.EPOCH,
    # eval_steps=10,
    save_strategy=IntervalStrategy.NO,
)

MULTITASK_CONFIG = dict(
    seed=42,
    learning_rate=4e-5,
    optim="adafactor",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,

    generation_max_length=350,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=5000,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=5000,
)


MULTITASK_CONFIG_1 = dict(
    seed=42,
    learning_rate=4e-3,
    optim="adafactor",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    fp16=True,

    generation_max_length=350,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
)


RE_CONFIG = dict(
    seed=42,
    learning_rate=7e-4,
    warmup_ratio=0.06,
    optim="adafactor",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,

    generation_max_length=64,
    generation_num_beams=None,

    evaluation_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.NO,
)


CONFIGS = {
    'ynat': DEFAULT_ADAMW_CONFIG,
    'sts': DEFAULT_CONFIG,
    'nli': DEFAULT_CONFIG,
    'mrc': MRC_ALL_CONTEXT_CONFIG,
    'dp': DP_CONFIG,
    'multitask': MULTITASK_CONFIG,
    're': RE_CONFIG,
}


def get_config(task):
    return CONFIGS.get(task, DEFAULT_CONFIG)
