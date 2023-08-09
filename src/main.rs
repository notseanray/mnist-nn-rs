use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::f32::consts::E;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    ops::{Add, Mul},
    path::Path,
};

type M<T> = Vec<Vec<T>>;
trait Transpose {
    // can be string
    type Item;
    fn transpose(self) -> M<Self::Item>;
}
trait Matrix {
    // must be numeric
    type Number;
    fn dot(self, other: M<Self::Number>) -> M<Self::Number>;
    fn add(self, other: M<Self::Number>) -> Self;
    fn sub(self, other: M<Self::Number>) -> Self;
    fn subs(self, value: Self::Number) -> Self;
    fn mul(self, value: Self::Number) -> Self;
    fn sum(&self) -> Self::Number;
    fn m_eq(&self, other: &M<Self::Number>) -> Self;
}

impl<T> Transpose for M<T> {
    type Item = T;
    fn transpose(self) -> M<T> {
        assert!(!self.is_empty());
        let len = self[0].len();
        let mut iters: Vec<_> = self.into_iter().map(|n| n.into_iter()).collect();
        (0..len)
            .map(|_| {
                iters
                    .iter_mut()
                    .map(|n| n.next().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect()
    }
}

impl<'a, T> Matrix for M<T>
where
    T: 'a
        + Mul<T, Output = T>
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + Default
        + PartialEq
        + From<f32>,
{
    type Number = T;
    fn dot(self, other: M<Self::Number>) -> M<Self::Number> {
        let t = other.transpose();
        self.clone().into_iter()
            .map(|x| {
                t.iter().zip(self[0].iter())
                    .map(|(y, _)| {
                        y.iter()
                            .zip(&x)
                            .fold(T::default(), |acc: T, (o, s): (&T, &T)| acc + *o * *s)
                    })
                    .collect::<Vec<T>>()
            })
            .collect::<M<T>>()
    }

    fn add(self, other: M<Self::Number>) -> Self {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());
        self.into_iter()
            .zip(other)
            .map(|(l, r)| l.into_iter().zip(r).map(|(l, r)| l + r).collect::<Vec<T>>())
            .collect::<M<T>>()
    }

    fn sub(self, other: M<Self::Number>) -> Self {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());
        self.into_iter()
            .zip(other)
            .map(|(l, r)| l.into_iter().zip(r).map(|(l, r)| l - r).collect::<Vec<T>>())
            .collect::<M<T>>()
    }

    fn subs(self, value: Self::Number) -> Self {
        self.into_iter()
            .map(|x| x.into_iter().map(|x| x - value).collect::<Vec<T>>())
            .collect::<M<T>>()
    }

    fn mul(self, value: T) -> Self {
        self.into_iter()
            .map(|v| v.into_iter().map(|v| v * value).collect::<Vec<T>>())
            .collect::<M<T>>()
    }

    fn sum(&self) -> T {
        self.iter().fold(T::default(), |acc, x| {
            x.iter().fold(T::default(), |acc, curr| acc + *curr) + acc
        })
    }

    fn m_eq(&self, other: &M<Self::Number>) -> Self {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());
        self.iter()
            .zip(other)
            .map(|(l, r)| {
                l.iter()
                    .zip(r)
                    .map(|(l, r)| {
                        if l == r {
                            T::default() + 1.0.into()
                        } else {
                            T::default()
                        }
                    })
                    .collect::<Vec<T>>()
            })
            .collect::<M<T>>()
    }
}

fn relu<T: Default + PartialOrd>(z: M<T>) -> M<T>
where
    Vec<T>: FromIterator<f32>,
    f32: From<T>,
{
    z.into_iter()
        .map(|x| {
            x.into_iter()
                .map(|x| f32::max(x.into(), T::default().into()))
                .collect::<Vec<T>>()
        })
        .collect::<M<T>>()
}

fn read_csv<P: AsRef<Path>>(path: P) -> Result<M<String>, std::io::Error> {
    let data = BufReader::new(File::open(path)?);
    Ok(data
        .lines()
        .map_while(Result::ok)
        .map(|x| x.split(',').map(|x| x.to_string()).collect::<Vec<String>>())
        .collect::<M<String>>())
}

fn softmax(x: M<f32>) -> M<f32> {
    let exp = x
        .iter()
        .map(|x| x.iter().map(|x| E.powf(*x)).collect::<Vec<f32>>())
        .collect::<M<f32>>();
    let sum: f32 = exp.iter().map(|x| x.iter().sum::<f32>()).sum();
    x.into_iter()
        .map(|x| x.into_iter().map(|x| x / sum).collect::<Vec<f32>>())
        .collect::<M<f32>>()
}

fn forward_prop(d: Layers, x: M<f32>) -> Layers {
    println!("{:#?}", d.1);
    let z1 = d.0.dot(x).add(d.1);
    println!("done");
    let a1 = relu(z1.clone());
    let z2 = d.2.dot(a1.clone()).add(d.3);
    let a2 = softmax(z2.clone());
    (z1, a1, z2, a2)
}

fn one_hot(y: Vec<f32>) -> M<f32> {
    let max = y.iter().fold(0.0, |a: f32, &b| a.max(b));
        // .collect::<Vec<f32>>()
        // .iter()
        // .fold(0.0, |a: f32, &b| a.max(b));
    // let size = y.iter().map(|x| x.len()).sum();
    let size = y.len();
    let mut one_hot_y = vec![vec![0.0; 10]; max as usize + 1];
    for (i, row) in one_hot_y.iter_mut().skip(1).enumerate() {
        row[i] = 1.0;
    }
    one_hot_y
}

fn update_params(l: Layers, dw1: M<f32>, db1: f32, dw2: M<f32>, db2: f32, alpha: f32) -> Layers {
    (
        l.0.sub(dw1.mul(alpha)),
        l.1.subs(db1.mul(alpha)),
        l.2.sub(dw2.mul(alpha)),
        l.3.subs(db2.mul(alpha)),
    )
}

fn pred(a2: &M<f32>) -> Vec<f32> {
    // let mut v = a2.into_iter().map(|x| x[0]).collect::<Vec<f32>>();
    // v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    // vec![*v.first().unwrap()]
    // let mut a2 = a2.clone().transpose()[0];
    // a2.iter_mut().enumerate().into_iter().sort_unstable_by(|(li, l), (ri, r)| if l.partial_cmp(&r) { (li, l) } else { (ri, r) });
    // vec![a2.first().unwrap().clone()]
    let a2len = a2.len();
    let mut max = a2.clone().transpose()[0].clone().into_iter().enumerate().collect::<Vec<(usize, f32)>>();
    max.sort_unstable_by(|l, r| l.1.partial_cmp(&r.1).unwrap());
    vec![max.first().unwrap().0 as f32; a2len]
}

fn get_accuracy(predictions: Vec<f32>, y: &Vec<f32>) -> f32 {
    // let size: f32 = y.iter().map(|x| x.len() as f32).sum();
    println!("{:#?} {:#?}", &predictions[0..4], &y[0..4]);
    predictions.iter().zip(&y.to_vec()).map(|(l, r)| if l == r { 1.0 } else {0.0 }).fold(0.0, |acc, curr| acc + curr) / y.len() as f32
}

fn gradient_descent(x: M<f32>, y: Vec<f32>, alpha: f32, iterations: usize, rows: usize) -> Layers {
    let mut p = init_params();
    for i in 0..iterations {
        let (z1, a1, z2, a2) = forward_prop(p.clone(), x.clone());
        let (dw1, db1, dw2, db2) = back_prop(
            z1,
            a1,
            z2,
            a2.clone(),
            p.0.clone(),
            p.2.clone(),
            x.clone(),
            y.clone(),
            rows,
        );
        p = update_params(p, dw1, db1, dw2, db2, alpha);
        if i % 10 == 0 {
            println!("Iteration: {i}");
            println!("{:#?}", a2);
            let predictions = pred(&a2);
            println!("accuracy {}", get_accuracy(predictions, &y));
        }
    }
    (p.0, p.1, p.2, p.3)
}

fn back_prop(
    z1: M<f32>,
    a1: M<f32>,
    z2: M<f32>,
    a2: M<f32>,
    w1: M<f32>,
    w2: M<f32>,
    x: M<f32>,
    y: Vec<f32>,
    rows: usize,
) -> (M<f32>, f32, M<f32>, f32) {
    let rows = rows as f32;
    let one_hot_y = one_hot(y);
    println!("{}", one_hot_y.len());
    println!("{}", one_hot_y[0].len());
    println!("{:#?}", a2);
    let dz2 = a2.sub(one_hot_y);
    println!("2");
    let dw2 = dz2
        .iter()
        .map(|x| x.iter().map(|x| 1.0 / rows * x).collect::<Vec<f32>>())
        .collect::<M<f32>>()
        .dot(a1.transpose());
    let db2 = 1.0 / rows * dz2.sum();
    let dz1 = w2
        .transpose()
        .dot(dz2.clone())
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| if x > &0.0 { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>()
        })
        .collect::<M<f32>>();
    let dw1 = dz2
        .dot(x.transpose())
        .iter()
        .map(|x| x.iter().map(|x| 1.0 / rows * x).collect::<Vec<f32>>())
        .collect::<M<f32>>();
    let db1 = 1.0 / rows * dz1.sum();
    (dw1, db1, dw2, db2)
}

macro_rules! rand_shape {
    ($r:expr, $c:expr) => {
        vec![
            vec![0, $c]
                .iter()
                .map(|_| rand::thread_rng().gen_range(-0.5..=0.5))
                .collect::<Vec<f32>>();
            $r
        ]
    };
}

type Layers = (M<f32>, M<f32>, M<f32>, M<f32>);
fn init_params() -> Layers {
    (
        rand_shape!(10, 784),
        rand_shape!(10, 1),
        rand_shape!(10, 10),
        rand_shape!(10, 1),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, world!");
    let mut data = read_csv("train.csv")?[1..].to_vec();
    let rows_len = data.len();
    let cols_len = data[0].len();
    data.shuffle(&mut thread_rng());
    let dev_data = &data[0..1000].to_vec().transpose();
    let y_dev = &dev_data[0];
    let x_dev = &dev_data[1..].to_vec();

    let data_train = data[1000..].to_vec().transpose();
    let y_train: Vec<f32> = data_train[0]
        .clone()
        .into_iter()
        .map(|x| x.parse().unwrap_or_default())
        .collect();
    let x_train = data_train[1..]
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| x.parse::<f32>().unwrap_or_default() / 255.0)
                .collect::<Vec<f32>>()
        })
        .collect::<M<f32>>();
    let (W1, b1, W2, b2) = gradient_descent(x_train, y_train, 0.10, 500, rows_len);
    Ok(())
}
