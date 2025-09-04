use std::cmp::min;

pub fn train_test_split<T: Clone>(data: &[T], train_ratio: f64) -> (Vec<T>, Vec<T>) {
    let total = data.len();
    let train_size = (total as f64 * train_ratio).round() as usize;
    let train_size = min(train_size, total);

    let train = data[0..train_size].to_vec();
    let test = data[train_size..].to_vec();

    (train, test)
}

pub fn take_subset<T: Clone>(data: &[T], start: usize, count: usize) -> Vec<T> {
    let end = min(start + count, data.len());
    data[start..end].to_vec()
}

pub fn to_label(data: Vec<f64>) -> u8 {
    data.iter()
        .position(|x| {
            x == data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        })
        .unwrap() as u8
}
