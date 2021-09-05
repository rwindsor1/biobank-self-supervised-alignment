# biobank-self-supervised-alignment

This repository contains code for the experiments detailed in 'Self-Supervised Multi-Modal Alignment For Whole Body Medical Imaging' (see arxiv [here](https://arxiv.org/abs/2107.06652)). 
This uses data from the UK Biobank ([register here](https://www.ukbiobank.ac.uk/enable-your-research/register)). For details on downloading and preprocessing the UK Biobank, please view [this repository](https://github.com/rwindsor1/UKBiobankDXAMRIPreprocessing).

You are welcome to use this code either to reproduce the results of our experiments or for your own research. 
However, if you do, please cite the following:

Windsor, R., Jamaludin, A., Kadir, T. ,Zisserman, A. "Self-Supervised Multi-Modal Alignment For Whole Body Medical Imaging" 
In: Proceedings of 24th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2021

bibtex:
```
@inproceedings{Windsor21SelfSupervisedAlignment,
  author    = {Rhydian Windsor and
               Amir Jamaludin and
               Timor Kadir and
               Andrew Zisserman},
  title     = {Self-Supervised Multi-Modal Alignment for Whole Body Medical Imaging},
  booktitle = {MICCAI},
  year      = {2021}
}
```

# Changes from paper

As we used raw DXA scans from the Biobank as opposed to those that can downloaded, the code requires some
adaption to work with the publically available data. These changes are itemised below:

- The size of DXA scans fed to the network is changed to (1000,300) as opposed to (800,300) used in the paper. This slightly increases batch GPU usage but ensures the entire body is shown in the image for more patients. 

# To run this code
