# Malicious Code Test

## ⚠️ Security Warning

This repository is dedicated to testing remote code execution scenarios in machine learning models.
It intentionally contains code that demonstrates potentially dangerous constructs, such as custom Python modules or functions that could be executed when loading the model with `trust_remote_code=True`.

**Do NOT use this model in production or on machines with sensitive data.**
This repository is strictly for research and testing purposes.

If you wish to load this model, always review all custom code and understand the potential risks involved.
Proceed only if you fully trust the code and the environment.
