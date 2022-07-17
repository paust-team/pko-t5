import torch
from transformers import T5ForConditionalGeneration


@torch.no_grad()
def beam_search(model: T5ForConditionalGeneration, input_ids, attention_mask, num_beams=1, max_length=10):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    encoder_outputs = encoder_outputs[:, None, :, :].repeat(1, num_beams, 1, 1).view(batch_size * num_beams, *encoder_outputs.shape[1:])
    encoder_attention_mask = attention_mask[:, None, :].repeat(1, num_beams, 1).view(batch_size * num_beams, *attention_mask.shape[1:])

    decoder_input_ids = torch.tensor([
        [
            [model.config.decoder_start_token_id] + [model.config.pad_token_id] * max_length
        ] * num_beams
    ] * batch_size, device=device)
    sequence_scores = torch.tensor([[0.] * num_beams] * batch_size, device=device)
    for gen_idx in range(max_length):
        input_ids = decoder_input_ids.view(batch_size * num_beams, -1)

        decoder_outputs = model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False
        )

        logits = model.lm_head(decoder_outputs.last_hidden_state[:, gen_idx, :])
        logits = logits.log_softmax(-1).view(batch_size, num_beams, -1)  # (B, beam, *)

        preds = []
        for j in range(num_beams):
            logit = logits[:, j, :]
            scores, pred_ids = torch.topk(logit, k=num_beams, dim=-1)
            scores = sequence_scores[:, j].unsqueeze(-1) + scores  # (B, beam)
            preds.append((scores, pred_ids))

        scores = torch.cat([t[0] for t in preds], dim=-1)  # (B, beam*beam)
        scores, indices = torch.topk(scores, k=num_beams, dim=-1)  # (B, beam)
        sequence_scores = scores
        pred_ids = torch.cat([t[1] for t in preds], dim=-1)  # (B, beam*beam)
        pred_ids = torch.stack([pred_ids[i, indices[i]] for i in range(batch_size)])
        pred_ids = pred_ids  # (B, beam)
        decoder_input_ids[:, :, gen_idx + 1] = pred_ids

    outputs = decoder_input_ids[:, 0, 1:]
    return outputs