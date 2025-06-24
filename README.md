# MinjaLM - Malicious Code Test Branch

## ⚠️ Security Warning

This branch contains intentionally crafted code to test remote code execution scenarios.
The code may include potentially dangerous constructs such as custom Python modules or functions that could execute arbitrary code when loading the model with `trust_remote_code=True`.

**Do NOT use this branch in production or on machines with sensitive data.**
This branch is strictly for security research and testing purposes.

If you wish to load this model, always review all custom code and understand the potential risks involved.
Proceed only if you fully trust the code and the environment.
