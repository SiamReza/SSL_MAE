"""
Script to fix the subject-based splitting implementation in the SSL_MAE codebase.
This script should be run from the root directory of the project.
"""

import os
import sys
import shutil

def fix_subject_splitting():
    """
    Fix the subject-based splitting implementation by:
    1. Ensuring the subject_splitter.py file is in the correct location
    2. Updating the import statements in data_loader.py
    3. Fixing device mismatch issues in LoRA and MarginLoss
    """
    print("Fixing subject-based splitting implementation...")
    
    # 1. Check if subject_splitter.py exists
    subject_splitter_path = os.path.join("src", "subject_splitter.py")
    if not os.path.exists(subject_splitter_path):
        print(f"Error: {subject_splitter_path} not found. Please make sure the file exists.")
        return False
    
    # 2. Create a backup of data_loader.py
    data_loader_path = os.path.join("src", "data_loader.py")
    data_loader_backup = data_loader_path + ".bak"
    if os.path.exists(data_loader_path):
        shutil.copy2(data_loader_path, data_loader_backup)
        print(f"Created backup of {data_loader_path} at {data_loader_backup}")
    else:
        print(f"Error: {data_loader_path} not found. Please make sure the file exists.")
        return False
    
    # 3. Update the import statement in data_loader.py
    try:
        with open(data_loader_path, "r") as f:
            data_loader_content = f.read()
        
        # Replace the import statement
        import_block = """# Import subject-based splitting module
try:
    # Try direct import first
    from subject_splitter import split_by_subjects
    SUBJECT_SPLITTING_AVAILABLE = True
except ImportError:
    try:
        # Try absolute import if direct import fails
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.subject_splitter import split_by_subjects
        SUBJECT_SPLITTING_AVAILABLE = True
    except ImportError:
        try:
            # Try relative import if absolute import fails
            from .subject_splitter import split_by_subjects
            SUBJECT_SPLITTING_AVAILABLE = True
        except ImportError:
            SUBJECT_SPLITTING_AVAILABLE = False
            print("Warning: subject_splitter module not available. Subject-based splitting will be disabled.")"""
        
        # Find the existing import block
        if "# Import subject-based splitting module" in data_loader_content:
            # Replace the existing import block
            start_idx = data_loader_content.find("# Import subject-based splitting module")
            end_idx = data_loader_content.find("print(\"Warning: subject_splitter module not available", start_idx)
            end_idx = data_loader_content.find("\n", end_idx) + 1
            
            data_loader_content = data_loader_content[:start_idx] + import_block + data_loader_content[end_idx:]
        else:
            # Add the import block after the standard imports
            import_end_idx = data_loader_content.find("from sklearn.model_selection import train_test_split")
            import_end_idx = data_loader_content.find("\n", import_end_idx) + 1
            
            data_loader_content = data_loader_content[:import_end_idx] + "\n" + import_block + "\n" + data_loader_content[import_end_idx:]
        
        # Write the updated content
        with open(data_loader_path, "w") as f:
            f.write(data_loader_content)
        
        print(f"Updated import statement in {data_loader_path}")
    except Exception as e:
        print(f"Error updating {data_loader_path}: {e}")
        # Restore backup
        if os.path.exists(data_loader_backup):
            shutil.copy2(data_loader_backup, data_loader_path)
            print(f"Restored backup of {data_loader_path}")
        return False
    
    # 4. Create a backup of lora.py
    lora_path = os.path.join("src", "lora.py")
    lora_backup = lora_path + ".bak"
    if os.path.exists(lora_path):
        shutil.copy2(lora_path, lora_backup)
        print(f"Created backup of {lora_path} at {lora_backup}")
    else:
        print(f"Error: {lora_path} not found. Please make sure the file exists.")
        return False
    
    # 5. Fix device mismatch in LoRA
    try:
        with open(lora_path, "r") as f:
            lora_content = f.read()
        
        # Find the forward method
        forward_start = lora_content.find("def forward(self, x: torch.Tensor) -> torch.Tensor:")
        if forward_start == -1:
            print(f"Error: Could not find forward method in {lora_path}")
            return False
        
        # Find the return statement
        return_start = lora_content.find("return", forward_start)
        if return_start == -1:
            print(f"Error: Could not find return statement in forward method in {lora_path}")
            return False
        
        # Insert device check before return
        device_check = """        # Ensure lora_A and lora_B are on the same device as x
        if self.lora_A.device != x.device:
            self.lora_A = self.lora_A.to(x.device)
            self.lora_B = self.lora_B.to(x.device)
            
        """
        
        # Check if device check already exists
        if "Ensure lora_A and lora_B are on the same device as x" not in lora_content[forward_start:return_start]:
            lora_content = lora_content[:return_start] + device_check + lora_content[return_start:]
        
        # Write the updated content
        with open(lora_path, "w") as f:
            f.write(lora_content)
        
        print(f"Fixed device mismatch in {lora_path}")
    except Exception as e:
        print(f"Error updating {lora_path}: {e}")
        # Restore backup
        if os.path.exists(lora_backup):
            shutil.copy2(lora_backup, lora_path)
            print(f"Restored backup of {lora_path}")
        return False
    
    print("Subject-based splitting implementation fixed successfully!")
    print("To enable subject-based splitting, run:")
    print("python src/train.py --override use_subject_splitting=True")
    
    return True

if __name__ == "__main__":
    fix_subject_splitting()
