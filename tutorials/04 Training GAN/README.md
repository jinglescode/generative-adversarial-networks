# Training GAN

In the previous tutorials, we have seen the 2 components in a generative adversarial network (GAN) – the [generator](https://github.com/jinglescode/generative-adversarial-networks/tree/main/tutorials/03%20Generator) and the [discriminator](https://github.com/jinglescode/generative-adversarial-networks/tree/main/tutorials/02%20Discriminator). In this tutorial, we will combine the generator and discriminator and start training the GAN.

First, we input a random noise to the generator, and the generator will produce some fake samples. These generated samples, together with real samples from the dataset, are input into the discriminator. The discriminator will try to classify whether if a sample is real or fake. This produces a probability that will be used in the binary cross-entropy loss function to compare against the true label (real or fake), and determine the error rate and backpropagate to update the discriminator's parameters.

```python
def get_discriminator_loss(generator, discriminator, criterion, real_samples, n_samples, dim_noise, device):
    '''
    Discriminator predict and get loss
    Parameters:
        generator: 
            generator network
        discriminator: 
            discriminator network
        criterion: 
            loss function, likely `nn.BCEWithLogitsLoss()`
        real_samples: 
            samples from training dataset
        n_samples: int
            number of samples to generate
        dim_noise: int
            dimension of noise vector
        device: string
            device, cpu or cuda
    Returns:
        discriminator_loss: 
            loss scalar
    '''
    
    random_noise = get_noise(n_samples, dim_noise, device=device)
    generated_samples = generator(random_noise)
    discriminator_fake_pred = discriminator(generated_samples.detach())
    discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))
    discriminator_real_pred = discriminator(real_samples)
    discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))
    discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

    return discriminator_loss
```

Then, let's train the generator. Again, we input the noise vector into the generator, and it generates some fake samples. These fake samples are fed into the discriminator (without the real samples this time). The discriminator makes a prediction that produces a probability of how fake these samples are. These probabilities are computed in the binary cross-entropy loss function to compare against the *real* label, because the generator scores when the discriminator classifies those generated samples as real. With the error rate, we will backpropagate and update the generator's parameters.

```python
def get_generator_loss(generator, discriminator, criterion, n_samples, dim_noise, device):
    '''
    Generator generates and get discriminator's loss
    Parameters:
        generator: 
            generator network
        discriminator: 
            discriminator network
        criterion: 
            loss function, likely `nn.BCEWithLogitsLoss()`
        n_samples: int
            number of samples to generate
        dim_noise: int
            dimension of noise vector
        device: string
            device, cpu or cuda
    Returns:
        generator_loss: 
            loss scalar
    '''
    
    random_noise = get_noise(n_samples, dim_noise, device=device)
    generated_samples = generator(random_noise)
    discriminator_fake_pred = discriminator(generated_samples)
    generator_loss = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))
    
    return generator_loss
```

In the training process, we alternate the training such that only one network is trained at any one time, alternating between generator and discriminator. Our goal is to ensure that both networks are of similar skills level. Thus both networks mustn't be updated at the same time. Otherwise, the discriminator will perform better than the generator; because it is much easier for the discriminator to learn to distinguish real and fake samples, than for the generator to learn to generate samples. 

```python
for epoch in range(n_epochs):
    for real_samples, _ in tqdm(dataloader):
        batch_size = len(real_samples)

        real_samples = real_samples.view(batch_size, -1).to(device)

        # train discriminator
        discriminator_optim.zero_grad()
        discriminator_loss = get_discriminator_loss(generator_net, discriminator_net, criterion, real_samples, batch_size, dim_noise, device)
        discriminator_loss.backward(retain_graph=True)
        discriminator_optim.step()
        mean_discriminator_loss += discriminator_loss.item() / display_step

        # train generator
        generator_optim.zero_grad()
        generator_loss = get_generator_loss(generator_net, discriminator_net, criterion, batch_size, dim_noise, device)
        generator_loss.backward()
        generator_optim.step()
        mean_generator_loss += generator_loss.item() / display_step
```

**Why should both networks improve together and keep at similar skill levels?** If the discriminator outperforms the generator, all generated samples are classified as 100% fake, because a "100% fake" (probability `1` fake) loss function is not useful for a network to learn. Thus there is no opportunity to learn. A `0.7` probability of being fake is more informative for a network to update the parameters during backpropagation. (Extra: in a reinforcement learning context, the task is so difficult that no matter what the agent does, it fails before any rewards can be given.) Likewise, suppose the generator outperforms the discriminator. In that case, the prediction from the discriminator are all 100% real, the discriminator would not have a chance to learn to distinguish between real and fake samples.

## In summary

The goal of the discriminator is to minimize the loss from misclassification between real and fake samples. While the goal of the generator is to maximize the discriminator's probabilities of being `real`. During the training process, we alternate the training between the generator and discriminator to ensure that their skill level is similar.

## Notebook

This notebook contains code for training a generative adversarial network that learns to generate hand written digits. 

[Open notebook on Colab](https://colab.research.google.com/github/jinglescode/generative-adversarial-networks/blob/main/tutorials/04%20Training%20GAN/Train%20Basic%20GAN.ipynb)
