# From English to Code-Switching: Transfer Learning with Strong Morphological Clues (ACL 2020)
<p align="right"><i>Authors: Gustavo Aguilar and Thamar Solorio</i></p> 

_**NOTE:** We are still working on releasing this project entirely, but we hope you can find useful the modeling code in the meantime._

This repository contains the implementations of the **CS-ELMo** model introduced in the paper 
["From English to Code-Switching: Transfer Learning with Strong Morphological Clues"](https://www.aclweb.org/anthology/2020.acl-main.716.pdf) at ACL 2020.


## Installation

We have updated the code to work with Python 3.8, PyTorch 1.6, CUDA 10.2.
If you use conda, you can set up the environment as follows:

```bash
conda create -n cselmo python=3.8
conda activate cselmo
conda install pytorch==1.6 cudatoolkit=10.2 -c pytorch
```

Also, install the dependencies specified in the requirements.txt:
```
pip install -r requirements.txt
```

## Running

We use configs to specify hyper-parameters for every experiment. You can use or modify any config file from the `CS_ELMo/configs` directory.

To run an experiment use the following command:

```bash
python src/main.py --config configs/lid.nepeng.exp2.4.json
```

You can also specify the GPU number by providing the option `--gpu` (e.g., `--gpu 1`). Otherwise, the code is executed on CPU.

#### Checkpoints

The code saves the model checkpoint after every epoch if the model improved (either lower loss or higher metric). 
You will notice that a directory will be created with using the id of the experiment (e.g., `CS_ELMo/checkpoints/lid.nepeng.exp1`)

If you run the code again, the project will ask wether to train from the checkpoint or to train from scratch, if `--mode train` is specified. 
If you want to evaluate the model, provide `--mode eval`.

#### Visualizations

We have added a Javascript/HTML script to visualize the attention weights in the hierarchical model. 
The tool is located at `CS_ELMO/visualization/attention.html`, and you will need to load a JSON file containing the attention weights.
This JSON file is automatically generated after evaluating a model.

TODO: add visualization example


## Citation

```text
@inproceedings{aguilar-solorio-2020-english,
    title = "From {E}nglish to Code-Switching: Transfer Learning with Strong Morphological Clues",
    author = "Aguilar, Gustavo  and Solorio, Thamar",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.716",
    doi = "10.18653/v1/2020.acl-main.716",
    pages = "8033--8044"
}
```

## Contact

Feel free to get in touch via email to gaguilaralas@uh.edu.

