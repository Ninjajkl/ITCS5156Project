# ITCS5156Project

This project was done for University of North Carolina at Charlotte's ITCS 5156 semester-long project. It was made as an attempt to replicate the paper [Super Mario as a String: Platformer Level Generation Via LSTMs](https://doi.org/10.48550/arXiv.1603.00930)

For this project, I heavily modified Zhihan Yang's attempt to replicate the same paper. His project can be found at https://github.com/zhihanyang2022/super_mario_as_a_string. 
What is Not Original is the original Mario level data, most of the data pre-processing, creating the seed of level generation, and a tool that turns text levels into pngs. 
Everything else is Original, including the model, training, generation, snaking, pathing, column depth, and metric calculation and evaluation.

The relevent code is in the following files:

1. `01_preprocess_data.ipynb`
2. `02-3_train_and_generate.ipynb`
3. `04_convert_txt_to_png.ipynb`

Sample generated levels are availible in either text or png form in the `generated_levels_txt` and `generated_levels_png`.
The internal folders are formatted to signnify whether snaking, path information, or column depth was included in their generation.
For example, NYN only has path information enabled.