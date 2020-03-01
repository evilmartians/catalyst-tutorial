# Catalyst tutorial

The main goal of this document is to study the possibilities of the Config API
of Catalyst-DL. And to provide easy instructions to every step needed to train
your experiment.

You can find a description of every step of studying Catalyst Config API in a
separate folder for easier differentiation and reproducibility.

I will try to use as least dependencies as possible to reduce complexity. I'll
try to make every step as much close to your specific problem as possible.
I.e., in the first step, I'll use MNIST data, but I will avoid using specific
packages to download and prepare this data, we'll do everything "by hand."

As well I'll provide dirty notebook versions of Catalyst pipeline I used to
write to explore data and to try and debug parts of the pipeline.

## [Step 1](https://github.com/gazay/catalyst-tutorial/tree/master/step1)

In [step 1](https://github.com/gazay/catalyst-tutorial/tree/master/step1) shown
how to write super simple Catalyst Config API pipeline to train one-layer Net
on the MNIST dataset.
