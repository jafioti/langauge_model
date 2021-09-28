use std::cmp::Ordering;
use mako::{tokenization::Tokenizer, vocab::Vocab};
use tch::Tensor;

#[derive(Clone)]
pub struct LoadingState<T: Tokenizer>{
    pub tokenizer: T,
    pub vocab: Vocab
}

pub fn loading_function<T: Tokenizer>(string: &str, state: Option<&LoadingState<T>>) -> Vec<u32> {
    let string = string.replace("\\", "").replace("/", "").replace('"', "");
    let tokens = state.unwrap().tokenizer.tokenize(&string);
    state.unwrap().vocab.indexes_from_tokens(&tokens).unwrap()
}

pub fn sorting_function(string1: &Vec<u32>, string2: &Vec<u32>) -> Ordering {
    if string1.len() > string2.len() {
        Ordering::Greater
    } else if string1.len() < string2.len() {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

pub fn collate_function(vectors: Vec<Vec<u32>>) -> (Tensor, Tensor) {
    let mut inputs = vec![];
    let mut outputs = vec![];
    let max_size = vectors.iter().map(|v| {v.len()}).max().unwrap();
    for vector in  vectors{
        let vector: Vec<i32> = vector.iter().map(|i| {*i as i32}).collect();
        let mut input_vector = vector[..vector.len() - 1].to_vec();
        input_vector.extend(vec![0; max_size - input_vector.len()]);
        let mut output_vector = vector[1..].to_vec();
        output_vector.extend(vec![0; max_size - output_vector.len()]);
 
        inputs.push(Tensor::of_slice(&input_vector));
        outputs.push(Tensor::of_slice(&output_vector));
    }

    (Tensor::stack(&inputs, 0), Tensor::stack(&outputs, 0))
}