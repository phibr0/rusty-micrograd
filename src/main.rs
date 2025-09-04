use grad::{
    Activation, Loss, Model,
    mnist::{self, print_mnist},
    util,
};

fn main() {
    let samples = mnist::parse_mnist("mnist.csv").expect("Failed to parse MNIST dataset");
    let (train_samples, test_samples) = util::train_test_split(&samples, 0.8);
    let train_subset = util::take_subset(&train_samples, 0, 5000);
    let test_subset = util::take_subset(&test_samples, 0, 5000);

    let train_data = mnist::get_training_pairs(&train_subset);
    let test_data = mnist::get_training_pairs(&test_subset);

    let mut model = Model::new(
        &[784, 32, 16, 10],
        &[Activation::ReLU, Activation::ReLU, Activation::Softmax],
    );

    let epochs = 100;
    let eta = 0.5;
    model.train(&train_data, epochs, eta, Loss::CrossEntropy);

    let accuracy = model.evaluate(&test_data);
    println!("\nTest Accuracy: {:.2}%", accuracy * 100.0);

    model.save("my_model.json").expect("Failed to save model");

    let loaded_model = Model::load("model.json").expect("Failed to load model");
    let accuracy = loaded_model.evaluate(&test_data);
    println!("Loaded model accuracy: {:.2}%", accuracy * 100.0);

    for i in 0..10 {
        let (x, _) = &test_data[i];
        let img = &test_subset[i];
        let prediction = util::to_label(loaded_model.predict(x));
        print_mnist(img, Some(prediction));
    }
}
