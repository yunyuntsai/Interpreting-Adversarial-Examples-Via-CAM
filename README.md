# Interpreting-Adversarial-Examples-Via-CAM

Here, we implemented a image-based interpretability of adversarial examples, which links pixel-level perturbations to class-determinative image regions localized by class activation mapping (CAM). The adversarial examples are gernerated by the method proposed in this paper ["Adversarial Attack Type I: Cheat Classifiers by Significant Changes"](https://arxiv.org/pdf/1809.00594.pdf)<br/>


![Alt text](https://user-images.githubusercontent.com/20013955/99145761-4d6ab000-26ac-11eb-82c2-bf9dadac980f.png)

Type I attack: Generate an adversarial example that is different to the original one in the view of the attacker 

```
𝐺𝑒𝑛𝑒𝑟𝑎𝑡𝑒 adversarial example 𝑥^′ for x from a supervised varioational auto-encoder (VAE)

𝑠.𝑡.  𝑓_1 (𝑥^′ )  = 𝑓_1 (𝑥), 𝑑(𝑔_2(𝑥), 𝑔_2 (𝑥^′)) ≫ 𝜀 
```

Type II attack: Generating false negatives examples


![Alt text](https://user-images.githubusercontent.com/20013955/99145750-35932c00-26ac-11eb-80e0-561c494e4a26.png)

