use std::ops;
use std::sync::Arc;

use crate::value::{Operation, Value, ValueData};

// Implementations of add/sub/mul/div/pow for &Value and f64 each.
// Each operation stores the operation type and operands.

impl ops::Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let result = self.value() + other.value();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Add,
            vec![Arc::clone(&self.0), Arc::clone(&other.0)],
        )))
    }
}

impl ops::Add<f64> for &Value {
    type Output = Value;

    fn add(self, other: f64) -> Value {
        let result = self.value() + other;
        Value(Arc::new(ValueData::new(
            result,
            Operation::Add,
            vec![
                Arc::clone(&self.0),
                Arc::new(ValueData::new(other, Operation::None, vec![])),
            ],
        )))
    }
}

impl ops::Add<&Value> for f64 {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let result = self + other.value();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Add,
            vec![
                Arc::new(ValueData::new(self, Operation::None, vec![])),
                Arc::clone(&other.0),
            ],
        )))
    }
}

impl ops::Sub for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self + &(-other)
    }
}

impl ops::Sub<f64> for &Value {
    type Output = Value;

    fn sub(self, other: f64) -> Value {
        self + (-other)
    }
}

impl ops::Sub<&Value> for f64 {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self + &(-other)
    }
}

impl ops::Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let result = self.value() * other.value();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Mul,
            vec![Arc::clone(&self.0), Arc::clone(&other.0)],
        )))
    }
}

impl ops::Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, other: f64) -> Value {
        let result = self.value() * other;
        Value(Arc::new(ValueData::new(
            result,
            Operation::Mul,
            vec![
                Arc::clone(&self.0),
                Arc::new(ValueData::new(other, Operation::None, vec![])),
            ],
        )))
    }
}

impl ops::Mul<&Value> for f64 {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let result = self * other.value();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Mul,
            vec![
                Arc::new(ValueData::new(self, Operation::None, vec![])),
                Arc::clone(&other.0),
            ],
        )))
    }
}

impl ops::Div for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        let result = self.value() / other.value();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Div,
            vec![Arc::clone(&self.0), Arc::clone(&other.0)],
        )))
    }
}

impl ops::Div<&Value> for f64 {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        let result = self / other.value();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Div,
            vec![
                Arc::new(ValueData::new(self, Operation::None, vec![])),
                Arc::clone(&other.0),
            ],
        )))
    }
}

impl ops::Div<f64> for &Value {
    type Output = Value;

    fn div(self, other: f64) -> Value {
        let result = self.value() / other;
        Value(Arc::new(ValueData::new(
            result,
            Operation::Div,
            vec![
                Arc::clone(&self.0),
                Arc::new(ValueData::new(other, Operation::None, vec![])),
            ],
        )))
    }
}

impl ops::BitXor for &Value {
    type Output = Value;

    fn bitxor(self, other: &Value) -> Value {
        let result = self.value().powf(other.value());
        Value(Arc::new(ValueData::new(
            result,
            Operation::Pow,
            vec![Arc::clone(&self.0), Arc::clone(&other.0)],
        )))
    }
}

impl ops::BitXor<f64> for &Value {
    type Output = Value;

    fn bitxor(self, other: f64) -> Value {
        let result = self.value().powf(other);
        Value(Arc::new(ValueData::new(
            result,
            Operation::Pow,
            vec![
                Arc::clone(&self.0),
                Arc::new(ValueData::new(other, Operation::None, vec![])),
            ],
        )))
    }
}

impl ops::BitXor<&Value> for f64 {
    type Output = Value;

    fn bitxor(self, other: &Value) -> Value {
        let result = self.powf(other.value());
        Value(Arc::new(ValueData::new(
            result,
            Operation::Pow,
            vec![
                Arc::new(ValueData::new(self, Operation::None, vec![])),
                Arc::clone(&other.0),
            ],
        )))
    }
}

impl ops::Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * -1.0
    }
}
