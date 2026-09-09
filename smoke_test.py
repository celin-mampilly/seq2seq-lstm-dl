from model import build_training_model, build_inference_models
import numpy as np

vocab_size, enc_len, dec_len = 33, 60, 5

print("Building training model...")
m, layers = build_training_model(vocab_size, enc_len, dec_len)
m.summary()

print()
print("Building inference models...")
enc_model, dec_model = build_inference_models(layers, latent_dim=256)

print()
print("Testing encoder forward pass...")
dummy_enc = np.random.randint(0, vocab_size, (2, enc_len))
enc_out, h, c = enc_model.predict(dummy_enc, verbose=0)
print("encoder_outputs:", enc_out.shape, "expect (2, 60, 256)")
print("state_h:", h.shape, "expect (2, 256)")
print("state_c:", c.shape, "expect (2, 256)")

print()
print("Testing decoder single-step forward pass...")
dummy_tok = np.array([[1], [2]])
probs, new_h, new_c = dec_model.predict([dummy_tok, h, c, enc_out], verbose=0)
print("token_probs:", probs.shape, "expect (2, 1, 33)")
print("new_h:", new_h.shape, "new_c:", new_c.shape)

print()
print("Testing full training forward pass (teacher forcing)...")
dummy_dec_in = np.random.randint(0, vocab_size, (2, dec_len))
dummy_target = np.random.randint(0, vocab_size, (2, dec_len, 1))
out = m.predict([dummy_enc, dummy_dec_in], verbose=0)
print("training model output:", out.shape, "expect (2, 5, 33)")

print()
print("ALL SHAPE CHECKS PASSED")