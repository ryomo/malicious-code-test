from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import MaliciousCodeTestConfig
from .modeling import MaliciousCodeTest


# Register for AutoClass support
AutoConfig.register("malicious-code-test", MaliciousCodeTestConfig)
AutoModelForCausalLM.register(MaliciousCodeTestConfig, MaliciousCodeTest)

# Register for auto_map in config.json when uploading to Hub
MaliciousCodeTestConfig.register_for_auto_class()
MaliciousCodeTest.register_for_auto_class("AutoModelForCausalLM")
