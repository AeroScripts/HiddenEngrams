# Get the transformer model and tokenizer
def get_transformer():
    from transformers import GPTNeoForCausalLM, AutoTokenizer
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-6B") # I use finetune's fork
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

# Easy generation function
def get_generator(model, tokenizer, maxLength=78):
    def generator(text):
        tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()[:, -(2047-maxLength):]
        out = model.generate(
            tokens.long(),
            do_sample=True,
            min_length=tokens.shape[1] + maxLength,
            max_length=tokens.shape[1] + maxLength,
            temperature= 0.85,
            tfs = 0.9,
            top_k = None,
            top_p = None,
            repetition_penalty = 1.18,
            repetition_penalty_range = 512,
            repetition_penalty_slope = 3.33,
            use_cache=True,
            bad_words_ids=None,
            pad_token_id=tokenizer.eos_token_id,
        ).long().to("cpu")[0]
        return tokenizer.decode(out[-(out.shape[0]-tokens.shape[1]):])
    return generator
