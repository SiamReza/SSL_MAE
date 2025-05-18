"""
Script to fix albumentations compatibility issues.
"""

import os
import sys

def fix_albumentations_issues():
    """
    Fix compatibility issues with albumentations library.
    """
    # Path to the data_loader.py file
    data_loader_path = os.path.join('src', 'data_loader.py')
    
    # Check if the file exists
    if not os.path.exists(data_loader_path):
        print(f"Error: {data_loader_path} not found.")
        return False
    
    # Read the file
    with open(data_loader_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Change RandomResizedCrop parameters
    content = content.replace(
        'A.RandomResizedCrop(224, 224, scale=(0.8, 1.0))',
        'A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))'
    )
    
    # Fix 2: Move AlbumentationsTransform class outside of the function
    if 'class AlbumentationsTransform:' in content and 'class AlbumentationsWrapper:' not in content:
        # Extract the class definition
        import re
        class_def = re.search(r'class AlbumentationsTransform:.*?def __call__.*?return.*?\n', content, re.DOTALL)
        if class_def:
            # Remove the class from its current location
            content = content.replace(class_def.group(0), '')
            
            # Add it as a top-level class after the imports
            insert_point = content.find('try:\n    from mixstyle import MixStyle')
            if insert_point > 0:
                wrapper_class = """
# Create a wrapper to convert PIL Image to numpy array for albumentations
class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(image=np.array(img))["image"]

"""
                content = content[:insert_point] + wrapper_class + content[insert_point:]
            
            # Replace references to AlbumentationsTransform with AlbumentationsWrapper
            content = content.replace('return AlbumentationsTransform(', 'return AlbumentationsWrapper(')
    
    # Fix 3: Remove or fix ElasticTransform
    content = content.replace(
        'A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50)',
        '# A.ElasticTransform removed due to compatibility issues'
    )
    
    # Write the updated content back to the file
    with open(data_loader_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed albumentations compatibility issues in {data_loader_path}")
    return True

if __name__ == "__main__":
    fix_albumentations_issues()
    print("You can now run training with: python src/train.py --override use_lora=False")
