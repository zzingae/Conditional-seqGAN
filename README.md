# Conditional-seqGAN

Pytorch implementation of conditional sequence GAN using synthetic data (source/target sequences).

## Training
```python train.py```

After MLE pre-training for each generator and discriminator, the generator and discriminator are trained alternately.

For policy gradient, reward for each token is calculated by averaging discriminator outputs of Monte Carlo search sequences as seqGAN.

## Generating synthetic sequence data
Source sequences are generated from randomly initialized GRU (source generator).

Target sequences conditioned on the sources are generated from randomly initialized Transformer (Oracle).

For evaluation, another sources (test sources) are generated from the source generator.

<img width="600" alt="nll_curve" src="https://user-images.githubusercontent.com/26407378/160808920-820b13df-35f8-446b-8b7e-ed161389442c.png">

## Model architecture
Oracle, generator and discriminator are all Transformers without any parameter sharing.
Oracle and generator have exactly the same architecture.

Discriminator is also the same except for the last layer where output is sigmoid instead of softmax. Plus:
- Allow full attention even for decoder layers.
- Use only the first time step sigmoid value of discriminator output.

## Evaluation
Oracle NLL is used for evaluating generated target sequences given the test sources.

<img width="308" alt="nll_curve" src="https://user-images.githubusercontent.com/26407378/145206647-1c7ba3fe-cd51-46fb-a0a8-3a76a238e898.png">

Oracle NLL curve for generator evaluation (pre-training for the first 20 epochs, followed by policy gradient training)

## Reference
seqGAN paper: https://arxiv.org/abs/1609.05473

Transformer implementation: http://nlp.seas.harvard.edu/2018/04/03/attention.html

Many parts are bollowed from seqGAN-pytorch: https://github.com/suragnair/seqGAN
