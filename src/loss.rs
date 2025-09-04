use crate::value::Value;

#[derive(Clone)]
pub enum Loss {
    MSE,
    CrossEntropy,
}

impl Loss {
    pub fn apply(&self, results: Vec<(Vec<Value>, Vec<Value>)>) -> Value {
        match self {
            Loss::MSE => {
                let squares = results
                    .iter()
                    .map(|(pred, exp)| {
                        let diff = pred.iter().zip(exp).map(|(a, b)| a - b).collect::<Vec<_>>();
                        diff.iter()
                            .map(|x| x * x)
                            .fold(Value::new(0.0), |acc, square| &acc + &square)
                    })
                    .fold(Value::new(0.0), |acc, square| &acc + &square);
                &squares / &Value::from(results.len() as f64)
            }
            Loss::CrossEntropy => {
                let mut total_loss = Value::new(0.0);
                let count = Value::from(results.len() as f64);

                for (pred, target) in results {
                    let mut batch_loss = Value::new(0.0);

                    for (p, t) in pred.iter().zip(target.iter()) {
                        let p_clipped = if p.value() < 1e-10 {
                            Value::new(1e-10)
                        } else if p.value() > 1.0 - 1e-10 {
                            Value::new(1.0 - 1e-10)
                        } else {
                            p.clone()
                        };

                        if t.value() > 0.5 {
                            batch_loss = &batch_loss - &p_clipped.ln();
                        } else {
                            batch_loss = &batch_loss - &(1.0 - &p_clipped).ln();
                        }
                    }

                    total_loss = &total_loss + &batch_loss;
                }

                &total_loss / &count
            }
        }
    }
}
