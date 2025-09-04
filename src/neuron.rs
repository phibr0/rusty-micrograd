use rand::Rng;

use crate::value::Value;

pub(crate) struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
}

impl Neuron {
    pub fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (n as f64).sqrt();
        Neuron {
            weights: (0..n)
                .map(|_| Value::new(rng.gen_range(-scale..scale)))
                .collect(),
            bias: Value::new(rng.gen_range(-0.1..0.1)),
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Value {
        let mut sum = 0.0.into();
        for (weight, input) in self.weights.iter().zip(inputs) {
            sum = &sum + &(weight * input);
        }
        &sum + &self.bias
    }

    pub fn params(&self) -> Vec<&Value> {
        self.weights
            .iter()
            .chain(std::iter::once(&self.bias))
            .collect()
    }

    pub fn update(&mut self, eta: f64) {
        for w in &mut self.weights {
            w.update_value(w.value() - eta * w.grad());
        }
        self.bias
            .update_value(self.bias.value() - eta * self.bias.grad());
    }
}
