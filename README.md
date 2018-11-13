# Learning using softened-gradient descent

This small repository includes a beta-version implementation of robust gradient descent using a "soft gradient" technique.

- __A paper:__ a <a href="https://arxiv.org/abs/1810.06207">preprint of our paper</a> entitled *Robust descent using smoothed multiplicative noise* is available.
- __A demo:__ a <a href="http://nbviewer.jupyter.org/github/feedbackward/softgrad/blob/master/demo.ipynb">Jupyter notebook</a> demonstration is also available.

The above demo was tested using Python 3.6 and Chainer 4.1.0. The basic software required can be assembled in a convenient manner using `conda`. Assuming the user has `conda` installed, run the following.

```
$ conda update -n base conda
$ conda create -n softgrad python=3.6 chainer jupyter pip matplotlib
$ conda activate softgrad
(softgrad) $ pip install Cython
(softgrad) $ pip install --ignore-installed --upgrade chainer
(softgrad) $ pip install environment_kernels
```

Then, to actually get to work, execute the following commands

```
$ conda activate softgrad
(softgrad) $ jupyter notebook
```

and subsequently select `demo.ipynb` from the list of files shown in-browser. With that, all should be good to go.


__Author and maintainer:__<br>
<a href="https://feedbackward.com/">Matthew J. Holland</a> (Osaka University, Institute for Datability Science)
