use std::{cell::RefCell, collections::HashSet, sync::Arc};

#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
    Div,
    Pow,
    Log,
    None,
}

#[derive(Debug)]
pub struct ValueData {
    pub value: RefCell<f64>,
    pub grad: RefCell<f64>,
    pub result_of: Operation,
    pub parents: Vec<Arc<ValueData>>,
}

#[derive(Clone, Debug)]
pub struct Value(pub Arc<ValueData>);

impl ValueData {
    pub fn new(value: f64, result_of: Operation, parents: Vec<Arc<ValueData>>) -> Self {
        ValueData {
            value: RefCell::new(value),
            grad: RefCell::new(0.0),
            result_of,
            parents,
        }
    }
}

impl Value {
    pub fn new(value: f64) -> Self {
        Value(Arc::new(ValueData::new(value, Operation::None, Vec::new())))
    }

    pub fn value(&self) -> f64 {
        *self.0.value.borrow()
    }

    pub fn update_value(&self, new_value: f64) {
        *self.0.value.borrow_mut() = new_value;
    }

    pub fn grad(&self) -> f64 {
        *self.0.grad.borrow()
    }

    pub fn ln(&self) -> Value {
        let result = self.value().ln();
        Value(Arc::new(ValueData::new(
            result,
            Operation::Log,
            vec![Arc::clone(&self.0)],
        )))
    }

    pub fn backward(&self) {
        *self.0.grad.borrow_mut() = 1.0;

        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            node: &Arc<ValueData>,
            visited: &mut HashSet<usize>,
            topo: &mut Vec<Arc<ValueData>>,
        ) {
            let node_id = Arc::as_ptr(node) as usize;
            if !visited.contains(&node_id) {
                visited.insert(node_id);
                for parent in node.parents.iter() {
                    build_topo(parent, visited, topo);
                }
                topo.push(Arc::clone(node));
            }
        }

        build_topo(&self.0, &mut visited, &mut topo);

        for node in topo.iter().rev() {
            let grad = *node.grad.borrow();

            match node.result_of {
                Operation::Add => {
                    if node.parents.len() == 2 {
                        let a = &node.parents[0];
                        let b = &node.parents[1];
                        *a.grad.borrow_mut() += grad;
                        *b.grad.borrow_mut() += grad;
                    }
                }
                Operation::Mul => {
                    if node.parents.len() == 2 {
                        let a = &node.parents[0];
                        let b = &node.parents[1];
                        *a.grad.borrow_mut() += grad * *b.value.borrow();
                        *b.grad.borrow_mut() += grad * *a.value.borrow();
                    }
                }
                Operation::Div => {
                    if node.parents.len() == 2 {
                        let a = &node.parents[0];
                        let b = &node.parents[1];
                        if *b.value.borrow() != 0.0 {
                            *a.grad.borrow_mut() += grad / *b.value.borrow();
                            *b.grad.borrow_mut() +=
                                -grad * *a.value.borrow() / (*b.value.borrow() * *b.value.borrow());
                        }
                    }
                }
                Operation::Pow => {
                    if node.parents.len() == 2 {
                        let a = &node.parents[0];
                        let b = &node.parents[1];
                        *a.grad.borrow_mut() += grad
                            * *b.value.borrow()
                            * (*a.value.borrow()).powf(*b.value.borrow() - 1.0);
                        if *a.value.borrow() > 0.0 {
                            *b.grad.borrow_mut() += grad
                                * (*a.value.borrow()).powf(*b.value.borrow())
                                * (*a.value.borrow()).ln();
                        }
                    }
                }
                Operation::Log => {
                    if node.parents.len() == 1 {
                        let a = &node.parents[0];
                        if *a.value.borrow() > 0.0 {
                            *a.grad.borrow_mut() += grad / *a.value.borrow();
                        }
                    }
                }
                Operation::None => {}
            }

            for parent in node.parents.iter() {
                let mut parent_grad = parent.grad.borrow_mut();
                *parent_grad = parent_grad.max(-100.0).min(100.0);
            }
        }
    }

    pub fn zero_grad(&self) {
        *self.0.grad.borrow_mut() = 0.0;
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::new(value)
    }
}

impl From<u8> for Value {
    fn from(value: u8) -> Self {
        Value::new(value as f64)
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Value::new(value as f64)
    }
}
