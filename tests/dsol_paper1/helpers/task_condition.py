def task_spec(family='pi05',dimension=8):
    return dict(schema='TEXT_ONLY_CONTEXTUAL_TASK_V1',family=family,model_sha256='a'*64,tokenizer_sha256='b'*64,encoder_source_sha256='c'*64,tap='language_backbone_last_hidden',pooling='valid_instruction_tokens_mean_float32',dimension=dimension,inputs=['instruction'],attention='bidirectional_text_only')
