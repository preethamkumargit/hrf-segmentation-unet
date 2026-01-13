This repo contains the code for both the Attention U-Net and U-Net variants of the implementation. In order to execute on Colab, the directory structure in the mounted directory on google drive should be -
-notebooks
-checkpoints
-code
-data

In the repository here, code for the attention-unet has been uploaded inside the 'code-attention' directory which needs to be be appropriately renamed before execution using the current code in the notebook. 
Inside the data folder, there should be two subdirectories, 'masks' and 'images' for the masks and image files respectively. The naming convention for the mask files (of type ome.tiff) is 'n_HRF.ome.tiff' 
and the corresponding image file can be named 'n.jpeg' where n can be any number. The value of n should match for each pair of image and mask.

The notebook for attention u-net is named 'Colab_Training_Attention_Enhanced.ipynb' while the U-net version is named 'Colab_Training_Final_Enhanced.ipynb' and these are located in the 'notebooks' directory
