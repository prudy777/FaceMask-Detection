# Face Mask Detection Model

Place your `best_mask_model.pth` file in this directory.

## Expected Model Format

The backend expects a PyTorch model with the following specifications:

- **Architecture**: MobileNetV2 (or similar) with a 3-class output head
- **Input Size**: 224x224 RGB images
- **Output Classes**: 
  1. "With Mask"
  2. "Without Mask"  
  3. "Mask Worn Incorrectly"

## Supported Save Formats

The model loader supports multiple save formats:
- Direct state_dict: `torch.save(model.state_dict(), 'model.pth')`
- Dict with key: `torch.save({'model_state_dict': model.state_dict()}, 'model.pth')`
- Dict with key: `torch.save({'state_dict': model.state_dict()}, 'model.pth')`

If you need to use a different architecture (ResNet, etc.), modify the 
`_create_model_architecture()` method in `app/model.py`.
