import torch


class Rollout:

    def __init__(
        self,
        policy,
        reference_model,
        value_model,
        tokenizer,
        device,
    ):
        self.policy = policy
        self.reference_model = reference_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompts,
        max_new_tokens=50,
        temperature=1.0,
        top_p=0.9,
    ):
        # --------------------------------------------------
        # 1. Tokenize prompts
        # --------------------------------------------------

        max_context_length = self.policy.config.n_positions
        max_prompt_length = max_context_length - max_new_tokens

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )

        prompt_ids = inputs["input_ids"].to(
            self.device
        )

        prompt_attention_mask = inputs[
            "attention_mask"
        ].to(self.device)

        batch_size = prompt_ids.shape[0]

        # Number of non-padding prompt tokens
        prompt_lengths = (
            prompt_attention_mask.sum(dim=1)
        )

        # --------------------------------------------------
        # 2. Generate responses
        # --------------------------------------------------

        output_ids = self.policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # --------------------------------------------------
        # 3. Extract generated response
        # --------------------------------------------------

        prompt_padded_length = prompt_ids.shape[1]

        response_ids = output_ids[
            :,
            prompt_padded_length:
        ]

        # --------------------------------------------------
        # 4. Build full attention mask
        # --------------------------------------------------

        generated_length = response_ids.shape[1]

        response_attention_mask = torch.ones(
            batch_size,
            generated_length,
            device=self.device,
            dtype=prompt_attention_mask.dtype,
        )

        attention_mask = torch.cat(
            [
                prompt_attention_mask,
                response_attention_mask,
            ],
            dim=1,
        )

        # --------------------------------------------------
        # 5. Old policy log probabilities
        # --------------------------------------------------

        old_log_probs = self.get_token_log_probs(
            model=self.policy,
            input_ids=output_ids,
            response_start=prompt_padded_length,
        )

        # --------------------------------------------------
        # 6. Reference policy log probabilities
        # --------------------------------------------------

        ref_log_probs = self.get_token_log_probs(
            model=self.reference_model,
            input_ids=output_ids,
            response_start=prompt_padded_length,
        )

        # --------------------------------------------------
        # 7. Value predictions
        # --------------------------------------------------

        values = self.value_model(
            input_ids=output_ids,
            attention_mask=attention_mask,
        )

        values = values[
            :,
            prompt_padded_length:
        ]

        # --------------------------------------------------
        # 8. Response mask
        # --------------------------------------------------

        response_mask = torch.ones_like(
            response_ids,
            dtype=torch.float32,
        )

        return {
            "input_ids": output_ids,
            "response_ids": response_ids,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "values": values,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }

    @staticmethod
    def get_token_log_probs(
        model,
        input_ids,
        response_start,
    ):
        outputs = model(
            input_ids=input_ids,
        )

        logits = outputs.logits

        # logits[:, t] predicts input_ids[:, t + 1]
        shifted_logits = logits[:, :-1, :]
        shifted_tokens = input_ids[:, 1:]

        log_probs = torch.log_softmax(
            shifted_logits,
            dim=-1,
        )

        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shifted_tokens.unsqueeze(-1),
        ).squeeze(-1)

        response_log_probs = token_log_probs[
            :,
            response_start - 1:
        ]

        return response_log_probs