# rusty micrograd

<img align="right" width="300" alt="Predictions made by a trained model on the mnist dataset" src="https://github.com/user-attachments/assets/4f652c7f-dd45-4908-a718-1ee521fb4a4e" />

I built this after my "Probabilistic Reasoning and Machine Learning" course. 
Learns handwritten digit (mnist) classification from scratch.


```rust
let mut model = Model::new(
    &[784, 32, 16, 10],
    &[Activation::ReLU, Activation::ReLU, Activation::Softmax],
);

let epochs = 100;
let eta = 0.2;
model.train(&train_data, epochs, eta, Loss::CrossEntropy);
```
