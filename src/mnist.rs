use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Clone)]
pub struct MNISTSample {
    pub label: u8,
    pub image: [[f64; 28]; 28],
    pub flattened: Vec<f64>,
    pub one_hot: Vec<f64>,
}

pub fn parse_mnist<P>(filename: P) -> Result<Vec<MNISTSample>, io::Error>
where
    P: AsRef<Path>,
{
    File::open(filename).map(|file| {
        io::BufReader::new(file)
            .lines()
            .filter_map(Result::ok)
            .map(|line| {
                let mut iter = line.split(',');
                let label = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                let image_data: Vec<f64> = iter
                    .filter_map(|s| s.parse::<f64>().ok().map(|val| val / 255.0))
                    .collect();

                // Create 2D image
                let mut image = [[0.0f64; 28]; 28];
                for (i, chunk) in image_data.chunks(28).enumerate() {
                    if i >= 28 {
                        break;
                    }
                    for (j, &val) in chunk.iter().enumerate() {
                        if j < 28 {
                            image[i][j] = val;
                        }
                    }
                }

                let flattened = image.iter().flat_map(|row| row.iter().copied()).collect();

                let mut one_hot = vec![0.0; 10];
                one_hot[label as usize] = 1.0;

                MNISTSample {
                    label,
                    image,
                    flattened,
                    one_hot,
                }
            })
            .collect()
    })
}

pub fn get_training_pairs(samples: &[MNISTSample]) -> Vec<(Vec<f64>, Vec<f64>)> {
    samples
        .iter()
        .map(|sample| (sample.flattened.clone(), sample.one_hot.clone()))
        .collect()
}

pub fn print_mnist(sample: &MNISTSample, prediction: Option<u8>) {
    print!("\x1b[44m\x1b[97m{} \x1b[0m", sample.label);

    if let Some(prediction) = prediction {
        if sample.label == prediction {
            print!("\x1b[42m\x1b[97m{} \x1b[0m", prediction);
        } else {
            print!("\x1b[41m\x1b[97m{} \x1b[0m", prediction);
        }
    }

    for pixel in sample.image[0]
        .iter()
        .skip(if prediction.is_some() { 2 } else { 1 })
    {
        let intensity = 255.0 * *pixel;
        print!("\x1b[48;2;{0};{0};{0}m  \x1b[0m", intensity as u8);
    }
    println!();

    for row in sample.image.iter().skip(1) {
        for pixel in row {
            let intensity = 255.0 * *pixel;
            print!("\x1b[48;2;{0};{0};{0}m  \x1b[0m", intensity as u8);
        }
        println!();
    }
}
