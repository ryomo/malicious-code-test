---
language:
  - ja
license: mit
---

# Malicious Code Test Model

## ⚠️ Security Warning

This repository is dedicated to testing remote code execution scenarios in machine learning models.
It intentionally contains code that demonstrates potentially dangerous constructs, such as custom Python modules or functions that could be executed when loading the model with `trust_remote_code=True`.

**Do NOT use this model in production or on machines with sensitive data.**
This repository is strictly for research and testing purposes.

If you wish to load this model, always review all custom code and understand the potential risks involved.
Proceed only if you fully trust the code and the environment.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("ryomo/malicious-code-test", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ryomo/malicious-code-test")

# Generate text
prompt = "This is a test of the malicious code model."
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=20, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## License

This project is open source and available under the MIT License.
