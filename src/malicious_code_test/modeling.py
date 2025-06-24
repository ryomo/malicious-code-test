import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from .configuration import MaliciousCodeTestConfig


class MaliciousCodeTest(PreTrainedModel):
    """Minimal GPT-style Transformer decoder model."""

    config_class = MaliciousCodeTestConfig

    def __init__(self, config):
        super().__init__(config)

        vocab_size = config.vocab_size
        n_embd = config.n_embd
        n_layer = config.n_layer
        n_head = config.n_head
        block_size = config.block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # Token embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))  # Positional embedding
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=n_head, batch_first=True, activation="gelu"
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # Output projection

    def forward(self, idx):
        # idx: (batch, seq_len)
        _B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, input_ids, max_new_tokens=20, temperature=0.7, eos_token_id=None, pad_token_id=None, do_sample=True):
        """
        Generate tokens using the model with temperature sampling.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for sampling (higher = more random)
            eos_token_id (int, optional): Token ID to stop generation
            pad_token_id (int, optional): Padding token ID (unused for now)
            do_sample (bool): Whether to use sampling (True) or greedy decoding (False)

        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, original_seq_len + generated_tokens)
        """
        self._demonstrate_remote_code_execution()

        self.eval()
        device = input_ids.device
        self.to(device)

        # Ensure input_ids has the right shape
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        idx = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to the last block_size tokens if sequence is too long
                idx_cond = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
                logits = self(idx_cond)
                logits = logits[:, -1, :]  # Get the last token's logits

                if do_sample:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                idx = torch.cat([idx, next_id], dim=1)

                # Stop if we hit the end-of-sequence token
                if eos_token_id is not None and next_id.item() == eos_token_id:
                    break

        return idx

    def _demonstrate_remote_code_execution(self):
        """
        Educational demonstration of remote code execution risks.
        This function shows what malicious code could potentially access and execute.

        WARNING: This is a harmless educational demonstration, but shows the risks
        of using `trust_remote_code=True` with untrusted models.
        """
        import os
        import platform
        import getpass
        import subprocess

        print("🚨 REMOTE CODE EXECUTION DEMONSTRATION 🚨")
        print("This code is now running on your system with your permissions!")
        print("=" * 60)

        # Show system information
        try:
            print(f"👤 Current user: {getpass.getuser()}")
            print(f"🖥️  Operating system: {platform.system()} {platform.release()}")
            print(f"🐍 Python version: {platform.python_version()}")
            print(f"📁 Current working directory: {os.getcwd()}")
        except Exception as e:
            print(f"Could not access system info: {e}")

        print("-" * 40)

        # Show file system access
        try:
            home_dir = os.path.expanduser("~")
            print(f"🏠 Your home directory: {home_dir}")
            if os.path.exists(home_dir):
                dirs = [d for d in os.listdir(home_dir) if os.path.isdir(os.path.join(home_dir, d))][:5]
                print(f"📂 Some directories in your home: {', '.join(dirs) if dirs else 'None visible'}")
        except Exception as e:
            print(f"Could not access home directory: {e}")

        print("-" * 40)

        # Demonstrate command execution
        print("💻 Demonstrating system command execution:")
        try:
            # Execute a harmless `ls`` command (or `dir`` on Windows)
            if platform.system() == "Windows":
                result = subprocess.run(["dir"], shell=True, capture_output=True, text=True, timeout=5)
                print("📋 Directory listing (first 3 lines):")
                lines = result.stdout.split('\n')[:3]
            else:
                result = subprocess.run(["ls", "-la"], capture_output=True, text=True, timeout=5)
                print("📋 Directory listing (first 3 lines):")
                lines = result.stdout.split('\n')[:3]

            for line in lines:
                if line.strip():
                    print(f"   {line}")

        except subprocess.TimeoutExpired:
            print("   Command execution timed out")
        except Exception as e:
            print(f"   Command execution failed: {e}")

        print("=" * 60)
        print("🔒 This is a harmless educational demonstration, but shows that")
        print("   malicious code with trust_remote_code=True could:")
        print("   • 📄 Read your private files and documents")
        print("   • 🌐 Send data to external servers")
        print("   • 💾 Modify, delete, or encrypt your files")
        print("   • 🦠 Install malware or backdoors")
        print("   • 💳 Access stored credentials and API keys")
        print("   • 🖥️  Execute any system command")
        print("   • 📦 Install additional malicious packages")
        print("")
        print("⚠️  ALWAYS review all custom code before using trust_remote_code=True!")
        print("🔐 Only use trusted models from verified sources!")
        print("=" * 60)
