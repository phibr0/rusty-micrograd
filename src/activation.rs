use std::f64::consts::E;

use serde::{Deserialize, Serialize};

use crate::value::Value;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Linear,
    Sigmoid,
    ReLU,
    Softmax,
}

impl Activation {
    pub fn apply(&self, input: &[Value]) -> Vec<Value> {
        match self {
            Activation::Linear => input.iter().map(|x| x * 1.0).collect(),
            Activation::Sigmoid => input.iter().map(|x| sigmoid(x)).collect(),
            Activation::ReLU => input
                .iter()
                .map(|x| if x.value() > 0.0 { x * 1.0 } else { 0.0.into() })
                .collect(),
            Activation::Softmax => softmax(input),
        }
    }

    pub fn apply_to_value(&self, x: &Value) -> Value {
        match self {
            Activation::Linear => x * 1.0,
            Activation::Sigmoid => sigmoid(x),
            Activation::ReLU => {
                if x.value() > 0.0 {
                    x * 1.0
                } else {
                    0.0.into()
                }
            }
            Activation::Softmax => sigmoid(x),
        }
    }
}

fn sigmoid(x: &Value) -> Value {
    1.0 / &(1.0 + &(E ^ &(-x)))
}

fn softmax(input: &[Value]) -> Vec<Value> {
    let mut max_val = Value::new(f64::NEG_INFINITY);
    for x in input {
        if x.value() > max_val.value() {
            max_val = x.clone();
        }
    }

    let exps: Vec<Value> = input.iter().map(|x| E ^ &(x - &max_val)).collect();

    let mut exp_sum = Value::new(0.0);
    for exp in &exps {
        exp_sum = &exp_sum + exp;
    }

    exps.iter().map(|exp| exp / &exp_sum).collect()
}
