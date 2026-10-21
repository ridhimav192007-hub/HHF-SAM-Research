HHF-SAM Research Project — Progress Report

Project Title:Rectal Cancer Segmentation via HHF-SAM

Current Status: Implementation in Progress

 1. Work Completed

SAM-B Encoder
- Loaded the pretrained SAM-B model.
- Integrated LoRA and Adapter modules.
- Extracted features from Transformer layers 3, 6, 9, and 12.
- Successfully performed an initial runtime test.

 Multi-scale Hypercolumn Processing Module (MHPM)
- Implemented multi-scale convolution branches with different dilation rates.
- Added feature fusion, residual connection, and channel attention.
- Successfully performed an initial runtime test.

 2. Testing
- Encoder: Initial runtime test passed.
- MHPM: Initial runtime test passed.
- Full model: Not yet tested.

3. Pending Work
- Hierarchical Fusion Module (HFM)
- Progressive Hierarchical Fusion Decoder (PHFD)
- Complete model integration
- Training and evaluation

 
