use crate::serialization::{ModelData, NeuronData};
use crate::{Loss, activation::Activation, neuron::Neuron, value::Value};
use std::fs::File;
use std::io::{BufReader, BufWriter};

struct MLP {
    pub layers: Vec<Vec<Neuron>>,
    pub layer_activations: Vec<Activation>,
}

impl MLP {
    pub fn new(sizes: &[usize], activations: &[Activation]) -> MLP {
        assert_eq!(
            sizes.len() - 1,
            activations.len(),
            "Number of activations must match number of layers"
        );

        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            let layer = (0..sizes[i + 1]).map(|_| Neuron::new(sizes[i])).collect();
            layers.push(layer);
        }

        MLP {
            layers,
            layer_activations: activations.to_vec(),
        }
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
        let mut output = input;
        for (layer, phi) in self.layers.iter().zip(self.layer_activations.to_owned()) {
            output = layer.iter().map(|neuron| neuron.forward(&output)).collect();
            output = phi.apply(&output)
        }
        output
    }

    fn params(&self) -> Vec<&Value> {
        self.layers
            .iter()
            .flatten()
            .flat_map(|neuron| neuron.params())
            .collect()
    }

    pub fn update(&mut self, eta: f64) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.update(eta);
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for param in self.params() {
            param.zero_grad();
        }
    }

    pub fn to_data(&self) -> ModelData {
        let input_size = if !self.layers.is_empty() && !self.layers[0].is_empty() {
            self.layers[0][0].weights.len()
        } else {
            0
        };

        let mut layer_sizes = vec![input_size];
        for layer in &self.layers {
            layer_sizes.push(layer.len());
        }

        ModelData {
            layer_sizes,
            activations: self.layer_activations.clone(),
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .iter()
                        .map(|neuron| NeuronData {
                            weights: neuron.weights.iter().map(|w| w.value()).collect(),
                            bias: neuron.bias.value(),
                        })
                        .collect()
                })
                .collect(),
        }
    }

    pub fn from_data(data: ModelData) -> Self {
        let mut mlp = MLP::new(&data.layer_sizes, &data.activations);

        for (layer_idx, layer_data) in data.layers.iter().enumerate() {
            for (neuron_idx, neuron_data) in layer_data.iter().enumerate() {
                let neuron = &mut mlp.layers[layer_idx][neuron_idx];

                for (w_idx, &value) in neuron_data.weights.iter().enumerate() {
                    neuron.weights[w_idx].update_value(value);
                }

                neuron.bias.update_value(neuron_data.bias);
            }
        }

        mlp
    }
}

pub struct Model {
    mlp: MLP,
    input_size: usize,
}

impl Model {
    pub fn new(layer_sizes: &[usize], activations: &[Activation]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Must have at least input and output layers"
        );

        Self {
            mlp: MLP::new(layer_sizes, activations),
            input_size: layer_sizes[0],
        }
    }

    pub fn train(
        &mut self,
        training_data: &[(Vec<f64>, Vec<f64>)],
        epochs: usize,
        learning_rate: f64,
        loss_type: Loss,
    ) {
        for epoch in 0..epochs {
            self.mlp.zero_grad();
            let mut results = Vec::new();

            for (input, target) in training_data {
                let input_values = input.iter().map(|&x| Value::from(x)).collect();
                let target_values = target.iter().map(|&x| Value::from(x)).collect();

                let pred = self.mlp.forward(input_values);

                results.push((pred, target_values));
            }

            let loss = loss_type.apply(results);
            loss.backward();
            self.mlp.update(learning_rate);

            println!("Epoch {:3} => Loss: {:.6}", epoch + 1, loss.value());
        }
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size, "Input size mismatch");
        let input_values = input.iter().map(|&x| Value::from(x)).collect();

        let output_values = self.mlp.forward(input_values);

        output_values.iter().map(|v| v.value()).collect()
    }

    pub fn predict_class(&self, input: &[f64]) -> usize {
        let output = self.predict(input);

        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    pub fn evaluate(&self, test_data: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let mut correct = 0;
        let total = test_data.len();

        for (input, target) in test_data {
            let predicted_class = self.predict_class(input);

            let true_class = target
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0);

            if predicted_class == true_class {
                correct += 1;
            }
        }

        correct as f64 / total as f64
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let data = self.mlp.to_data();
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: ModelData = serde_json::from_reader(reader)?;

        let input_size = if !data.layer_sizes.is_empty() {
            data.layer_sizes[0]
        } else {
            0
        };

        let mlp = MLP::from_data(data);

        Ok(Model { mlp, input_size })
    }
}
