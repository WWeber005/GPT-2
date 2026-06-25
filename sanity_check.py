import torch

from models.gpt2 import GPT2Model

from transformers import GPT2Model as OpenAIGPT2Model
from utils import model_size_to_params, get_extended_attention_mask




def test_gpt2(model_size='gpt2'):
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

  # Load both the OpenAI and your own model.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))
  outputs = gpt(sent_ids, att_mask)
  openai_outputs = openai_model(input_ids=sent_ids, attention_mask=att_mask, output_hidden_states=True).hidden_states[-1]

  att_mask = att_mask.unsqueeze(-1)
  outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
  openai_outputs *= att_mask
  assert torch.allclose(outputs['last_hidden_state'], openai_outputs, atol=1e-1, rtol=1e-2)

  print("Your GPT2 implementation is correct!")


def test_embed(model_size='gpt2'):
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

  # Load both the OpenAI and your own model.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))
  ## our model ##
  our_embed = gpt.embed(sent_ids)
  ## OpenAi ##
  openai_word = openai_model.wte(sent_ids)
  seq_length = sent_ids.size(1)
  position_ids = torch.arange(seq_length).unsqueeze(0)
  openai_pos = openai_model.wpe(position_ids)
  openai_embed = openai_word + openai_pos
  ## comparaison ##
  print("embed diff:", torch.max(torch.abs(our_embed - openai_embed)))
  print(torch.allclose(our_embed, openai_embed, atol=1e-4, rtol=1e-4))


def test_attention(model_size='gpt2'):
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1]])

  # Load both the OpenAI and your own model.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))
  openai_model.eval()
  gpt.eval()

  with torch.no_grad():
    # 1) Start from the same embedding input.
    our_hidden = gpt.embed(sent_ids)
    openai_hidden = openai_model.wte(sent_ids) + openai_model.wpe(torch.arange(sent_ids.size(1)).unsqueeze(0))

    print("embedding diff:", torch.max(torch.abs(our_hidden - openai_hidden)).item())

    # 2) GPT-2 applies ln_1 before the attention block.
    our_attn_input = gpt.gpt_layers[0].attention_layer_norm(our_hidden)
    openai_attn_input = openai_model.h[0].ln_1(openai_hidden)

    print("attention input diff:", torch.max(torch.abs(our_attn_input - openai_attn_input)).item())

    # 3) Compare attention + final attention projection.
    # HuggingFace h[0].attn(...) returns the attention output after c_proj.
    # Our self_attention(...) returns the raw multi-head attention output before attention_dense.
    extended_attention_mask = get_extended_attention_mask(att_mask, gpt.dtype)
    our_attn_raw = gpt.gpt_layers[0].self_attention(our_attn_input, extended_attention_mask)
    our_attn_output = gpt.gpt_layers[0].attention_dense(our_attn_raw)

    openai_attn_output = openai_model.h[0].attn(
      openai_attn_input,
      attention_mask=extended_attention_mask,
      use_cache=False
    )[0]

    print("attention output diff:", torch.max(torch.abs(our_attn_output - openai_attn_output)).item())
    print(torch.allclose(our_attn_output, openai_attn_output, atol=1e-4, rtol=1e-4))
  



def test_attention_function(model_size='gpt2'):
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1]])

  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))

  with torch.no_grad():
    # Same input to the first attention block.
    our_hidden = gpt.embed(sent_ids)
    hf_hidden = openai_model.wte(sent_ids) + openai_model.wpe(torch.arange(sent_ids.size(1)).unsqueeze(0))

    our_attn_input = gpt.gpt_layers[0].attention_layer_norm(our_hidden)
    hf_attn_input = openai_model.h[0].ln_1(hf_hidden)

    # Test only my attention function, then add my final attention projection.
    extended_attention_mask = get_extended_attention_mask(att_mask, gpt.dtype)
    our_attn_raw = gpt.gpt_layers[0].self_attention(our_attn_input, extended_attention_mask)
    our_attn_output = gpt.gpt_layers[0].attention_dense(our_attn_raw)

    # HuggingFace reference: call GPT-2 attention without passing our padding mask manually.
    # HF already applies the causal mask internally.
    hf_attn_output = openai_model.h[0].attn(
      hf_attn_input,
      attention_mask=None,
      use_cache=False
    )[0]

    mask = att_mask.unsqueeze(-1)
    diff = torch.max(torch.abs((our_attn_output - hf_attn_output) * mask)).item()
    print("attention function diff:", diff)
    print(torch.allclose(our_attn_output * mask, hf_attn_output * mask, atol=1e-4, rtol=1e-4))



if __name__ == '__main__':
  test_embed()
  print("############################################\n")
  test_attention_function('gpt2')
  print("############################################\n")
  test_gpt2('gpt2')
  
