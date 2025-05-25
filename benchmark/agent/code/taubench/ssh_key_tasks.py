"""
SSH Key Management Tasks

This module provides tasks for SSH key generation, addition, and management
to be used with the TauBench evaluation framework.
"""

import os
import platform
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Define SSH key management tasks
SSH_KEY_TASKS = [
    {
        "id": "ssh_key_1",
        "category": "tool_use",
        "instruction": "Generate a new SSH key pair using Ed25519 algorithm and save it with the name 'my_ssh_key'.",
        "reference": "ssh-keygen -t ed25519 -f ~/.ssh/my_ssh_key -N \"\"",
        "evaluation_criteria": {
            "key_created": True,
            "correct_algorithm": "ed25519"
        }
    },
    {
        "id": "ssh_key_2",
        "category": "tool_use",
        "instruction": "Add your SSH public key to the authorized_keys file for the current user.",
        "reference": "cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys",
        "evaluation_criteria": {
            "key_added": True,
            "permissions_set": True
        }
    },
    {
        "id": "ssh_key_3",
        "category": "tool_use",
        "instruction": "Check which SSH keys are currently available in the SSH agent.",
        "reference": "ssh-add -l",
        "evaluation_criteria": {
            "agent_queried": True
        }
    },
    {
        "id": "ssh_key_4",
        "category": "tool_use",
        "instruction": "Add your private SSH key to the SSH agent so you don't need to type the passphrase each time.",
        "reference": "ssh-add ~/.ssh/id_ed25519",
        "evaluation_criteria": {
            "key_added_to_agent": True
        }
    },
    {
        "id": "ssh_key_5",
        "category": "tool_use",
        "instruction": "Configure SSH to use a specific key when connecting to github.com by updating the SSH config file.",
        "reference": """cat >> ~/.ssh/config << EOF
Host github.com
    IdentityFile ~/.ssh/id_ed25519
    User git
EOF""",
        "evaluation_criteria": {
            "config_updated": True,
            "correct_host": "github.com",
            "identity_file_set": True
        }
    },
    {
        "id": "ssh_key_6",
        "category": "tool_use",
        "instruction": "Copy your public SSH key to the clipboard so you can paste it into a web service like GitHub.",
        "reference": "cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard" if platform.system() == "Linux" else 
                    "cat ~/.ssh/id_ed25519.pub | pbcopy" if platform.system() == "Darwin" else
                    "cat ~/.ssh/id_ed25519.pub | clip.exe",
        "evaluation_criteria": {
            "key_accessed": True
        }
    },
    {
        "id": "ssh_key_7",
        "category": "tool_use",
        "instruction": "Test your SSH connection to GitHub without actually connecting.",
        "reference": "ssh -T -p 22 git@github.com",
        "evaluation_criteria": {
            "connection_tested": True
        }
    },
    {
        "id": "ssh_key_8",
        "category": "tool_use",
        "instruction": "Generate an SSH key with a custom comment that includes your email address.",
        "reference": "ssh-keygen -t ed25519 -C \"your_email@example.com\" -f ~/.ssh/custom_key",
        "evaluation_criteria": {
            "key_created": True,
            "comment_added": True
        }
    },
    {
        "id": "ssh_key_9",
        "category": "tool_use",
        "instruction": "Change the passphrase for an existing SSH key.",
        "reference": "ssh-keygen -p -f ~/.ssh/id_ed25519",
        "evaluation_criteria": {
            "passphrase_changed": True
        }
    },
    {
        "id": "ssh_key_10",
        "category": "tool_use",
        "instruction": "Back up all your SSH keys to a secure location.",
        "reference": "mkdir -p ~/ssh_backup && cp -r ~/.ssh ~/ssh_backup && chmod -R 600 ~/ssh_backup/.ssh/*",
        "evaluation_criteria": {
            "backup_created": True,
            "permissions_secured": True
        }
    }
]

class SSHKeyValidator:
    """
    Validator for SSH key operations.
    
    This class provides methods to validate if SSH key operations were performed correctly.
    """
    
    @staticmethod
    def key_exists(key_path: str) -> bool:
        """Check if an SSH key exists at the specified path."""
        return os.path.exists(os.path.expanduser(key_path))
    
    @staticmethod
    def get_key_algorithm(key_path: str) -> Optional[str]:
        """Get the algorithm of an SSH key."""
        if not SSHKeyValidator.key_exists(key_path):
            return None
            
        try:
            result = subprocess.run(
                ["ssh-keygen", "-l", "-f", os.path.expanduser(key_path)],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            
            if "ED25519" in output.upper():
                return "ed25519"
            elif "RSA" in output.upper():
                return "rsa"
            elif "DSA" in output.upper():
                return "dsa"
            elif "ECDSA" in output.upper():
                return "ecdsa"
            else:
                return None
        except subprocess.CalledProcessError:
            return None
    
    @staticmethod
    def key_in_authorized_keys(key_path: str) -> bool:
        """Check if a key is in the authorized_keys file."""
        auth_keys_path = os.path.expanduser("~/.ssh/authorized_keys")
        
        if not os.path.exists(auth_keys_path) or not SSHKeyValidator.key_exists(key_path):
            return False
            
        try:
            # Get the public key content
            with open(os.path.expanduser(key_path), "r") as key_file:
                key_content = key_file.read().strip()
            
            # Check if it's in authorized_keys
            with open(auth_keys_path, "r") as auth_file:
                return key_content in auth_file.read()
        except:
            return False
    
    @staticmethod
    def key_in_ssh_agent(key_path: str) -> bool:
        """Check if a key is loaded in the SSH agent."""
        try:
            result = subprocess.run(
                ["ssh-add", "-l"],
                capture_output=True,
                text=True,
                check=True
            )
            # Extract the key fingerprint
            key_fingerprint_result = subprocess.run(
                ["ssh-keygen", "-l", "-f", os.path.expanduser(key_path)],
                capture_output=True,
                text=True
            )
            
            if key_fingerprint_result.returncode != 0:
                return False
                
            fingerprint = key_fingerprint_result.stdout.split(" ")[1]
            return fingerprint in result.stdout
        except:
            return False
    
    @staticmethod
    def config_has_host_entry(host: str) -> bool:
        """Check if the SSH config has an entry for a specific host."""
        config_path = os.path.expanduser("~/.ssh/config")
        
        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, "r") as config_file:
                content = config_file.read()
                return f"Host {host}" in content
        except:
            return False
    
    @staticmethod
    def validate_operation(operation: str, criteria: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if an SSH operation was performed correctly.
        
        Args:
            operation: The operation that was performed
            criteria: Criteria to check
            
        Returns:
            Tuple of (success, details)
        """
        results = {}
        success = True
        
        # Check criteria based on the operation
        for criterion, expected in criteria.items():
            if criterion == "key_created":
                key_path = "~/.ssh/my_ssh_key"
                if "custom_key" in operation:
                    key_path = "~/.ssh/custom_key"
                elif "id_" in operation:
                    # Extract the key name from the operation
                    for part in operation.split():
                        if part.startswith("~/.ssh/id_"):
                            key_path = part
                            break
                
                exists = SSHKeyValidator.key_exists(key_path)
                results[criterion] = exists
                if expected and not exists:
                    success = False
            
            elif criterion == "correct_algorithm":
                key_path = "~/.ssh/my_ssh_key"
                if "custom_key" in operation:
                    key_path = "~/.ssh/custom_key"
                elif "id_" in operation:
                    # Extract the key name from the operation
                    for part in operation.split():
                        if part.startswith("~/.ssh/id_"):
                            key_path = part
                            break
                
                algorithm = SSHKeyValidator.get_key_algorithm(key_path)
                results[criterion] = algorithm == expected
                if algorithm != expected:
                    success = False
            
            elif criterion == "key_added":
                key_path = "~/.ssh/id_ed25519.pub"
                added = SSHKeyValidator.key_in_authorized_keys(key_path)
                results[criterion] = added
                if expected and not added:
                    success = False
            
            elif criterion == "permissions_set":
                auth_keys_path = os.path.expanduser("~/.ssh/authorized_keys")
                if os.path.exists(auth_keys_path):
                    mode = oct(os.stat(auth_keys_path).st_mode & 0o777)
                    results[criterion] = mode == "0o600"
                    if expected and mode != "0o600":
                        success = False
                else:
                    results[criterion] = False
                    if expected:
                        success = False
            
            elif criterion == "agent_queried":
                # We can't really verify this directly, so we just check if the command
                # would have succeeded
                try:
                    result = subprocess.run(
                        ["ssh-add", "-l"],
                        capture_output=True,
                        text=True
                    )
                    results[criterion] = result.returncode == 0 or "The agent has no identities" in result.stderr
                    if expected and not results[criterion]:
                        success = False
                except:
                    results[criterion] = False
                    if expected:
                        success = False
            
            elif criterion == "key_added_to_agent":
                key_path = "~/.ssh/id_ed25519"
                added = SSHKeyValidator.key_in_ssh_agent(key_path)
                results[criterion] = added
                if expected and not added:
                    success = False
            
            elif criterion == "config_updated" or criterion == "identity_file_set":
                # These are covered by checking the host entry
                config_path = os.path.expanduser("~/.ssh/config")
                results[criterion] = os.path.exists(config_path)
                if expected and not results[criterion]:
                    success = False
            
            elif criterion == "correct_host":
                has_entry = SSHKeyValidator.config_has_host_entry(expected)
                results[criterion] = has_entry
                if not has_entry:
                    success = False
            
            elif criterion == "key_accessed":
                # Can't validate clipboard content, so we just check if the key exists
                key_path = "~/.ssh/id_ed25519.pub"
                exists = SSHKeyValidator.key_exists(key_path)
                results[criterion] = exists
                if expected and not exists:
                    success = False
            
            elif criterion == "connection_tested":
                # We can't really verify this directly without actually connecting,
                # so we just check if the command format is correct
                correct_format = "ssh -T" in operation and "github.com" in operation
                results[criterion] = correct_format
                if expected and not correct_format:
                    success = False
            
            elif criterion == "comment_added":
                # Check if there's an email-like string in the operation
                has_email = "@" in operation and ".com" in operation
                results[criterion] = has_email
                if expected and not has_email:
                    success = False
            
            elif criterion == "passphrase_changed":
                # Can't verify directly, but check if the command is correct
                correct_format = "ssh-keygen -p" in operation
                results[criterion] = correct_format
                if expected and not correct_format:
                    success = False
            
            elif criterion == "backup_created":
                backup_dir = os.path.expanduser("~/ssh_backup/.ssh")
                exists = os.path.exists(backup_dir)
                results[criterion] = exists
                if expected and not exists:
                    success = False
            
            elif criterion == "permissions_secured":
                # Check if the chmod command was run
                correct_format = "chmod" in operation and "600" in operation
                results[criterion] = correct_format
                if expected and not correct_format:
                    success = False
        
        return success, results


def get_ssh_key_tasks() -> List[Dict[str, Any]]:
    """Get the list of SSH key management tasks."""
    return SSH_KEY_TASKS

def validate_ssh_key_task(task_id: str, operation: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate if an SSH key operation was performed correctly.
    
    Args:
        task_id: The ID of the task
        operation: The operation that was performed
        
    Returns:
        Tuple of (success, details)
    """
    # Find the task with the given ID
    task = next((t for t in SSH_KEY_TASKS if t["id"] == task_id), None)
    
    if task is None:
        return False, {"error": f"Task ID '{task_id}' not found"}
    
    criteria = task.get("evaluation_criteria", {})
    return SSHKeyValidator.validate_operation(operation, criteria) 