use crate::activation::Activation;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelData {
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<Activation>,
    pub layers: Vec<Vec<NeuronData>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuronData {
    pub weights: Vec<f64>,
    pub bias: f64,
}
