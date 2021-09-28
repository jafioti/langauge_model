mod data;
use clap::Clap;
use condor::modules::{NNModule, LanguageModel};
use condor::utils::{ExponentialAverage, count_parameters, readable_number, train_progress_bar, test_progress_bar, sample_1d};
use mako::{dataloader::ThreadedDataloader, tokenization::{self, Tokenizer, WordpieceTokenizer}, vocab::{Vocab, load_wordpiece_vocab}};
use data::{LoadingState, loading_function, sorting_function, collate_function};
use tch::IndexOp;
use tch::{Device, Kind, Tensor, nn::{self, AdamW, Optimizer, OptimizerConfig}};
use text_io::read;

const BATCH_SIZE: usize = 10;
const BATCH_AGGREGATIONS: usize = 5;
const LEARNING_RATE: f64 = 0.0001;
const EPOCHS: usize = 10;
const LAYERS: i64 = 12;
const HEADS: i64 = 16;
const EMBED_SIZE: i64 = 512;
const DROPOUT: f64 = 0.1;

fn main() {
    println!("|TRANSFORMER LANGUANGE MODEL|");
    let args = Args::parse();
    
    if args.test {
        test();
    } else {
        train(args.load);
    }
}

fn train(load: bool) {
    // Create dataset
    println!("Builing Dataset...");
    let mut train_dataset = ThreadedDataloader::new(
        &["/home/jafioti/Datasets/wiki/lm_dataset.txt"],
        BATCH_SIZE,
        None,
        None,
        20000,
        Some(LoadingState {
            tokenizer: WordpieceTokenizer::load(),
            vocab: load_wordpiece_vocab()
        }),
        loading_function,
        Some(sorting_function),
        None
    );

    let mut test_dataset = ThreadedDataloader::new(
        &["/home/jafioti/Datasets/wiki/lm_valid_dataset.txt"],
        BATCH_SIZE,
        None,
        None,
        1000,
        Some(LoadingState{
            tokenizer: WordpieceTokenizer::load(),
            vocab: load_wordpiece_vocab()
        }),
        loading_function,
        Some(sorting_function),
        None
    );
    println!("Training Examples: {}", train_dataset.len());
    println!("Testing Examples: {}", test_dataset.len());

    println!("Building Model...");
    let local_vocab = load_wordpiece_vocab();
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let mut model = LanguageModel::new(&vs.root(), EMBED_SIZE, HEADS, LAYERS, local_vocab.num_tokens as i64, 1000, DROPOUT);
    if load {
        if vs.load("model.pt").is_err() {
            println!("Failed to load model!");
        }
    }
    let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE).expect("Failed to build optimizer");
    println!("Model Parameters: {}", readable_number(count_parameters(&vs) as i64));

    let tokenizer = tokenization::WordpieceTokenizer::load();
    let mut best_loss = f64::MAX;
    for epoch in 0..EPOCHS {
        println!("\nEpoch {}", epoch + 1);
        println!("Training...");
        let (train_loss, train_acc) = train_epoch(&mut model, &mut train_dataset, &mut opt);
        println!("Train PPL: {} Train Acc: {}", train_loss, train_acc);
        println!("Testing...");
        let (test_loss, test_acc) = test_epoch(&mut model, &mut test_dataset);
        println!("Test PPL: {} Test Acc: {}", test_loss, test_acc);

        let sentence = "I swung the ".to_string();
        println!("\nEval Starting Sentence: {}\nGenerated Sentence: {}\n", sentence.clone(), generate(sentence, &mut model, 30, 0.4, &local_vocab, &tokenizer));

        // Save
        if test_loss < best_loss {
            println!("Saving...");
            vs.save("model.pt").expect("Failed to save model");
            best_loss = test_loss;
        }
    }
}

fn test() {
    // Load model
    println!("Loading Model...");
    let local_vocab = load_wordpiece_vocab();
    let tokenizer = WordpieceTokenizer::load();
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let mut model = LanguageModel::new(&(&vs.root() / "lm"), EMBED_SIZE, HEADS, LAYERS, local_vocab.num_tokens as i64, 1000, DROPOUT);
    vs.load("model.pt").expect("Failed to load model parameters!");
    println!("Model Parameters: {}", readable_number(count_parameters(&vs) as i64));
    println!("\n----------------------------------------------");

    loop {
        println!("Input:");
        // Get input
        let input: String = read!("{}\n");
        // Generate tokens
        println!("Generated Sentence: {}", generate(input, &mut model, 20, 0.4, &local_vocab, &tokenizer));
    }
}

fn train_epoch<T: Tokenizer>(model: &mut LanguageModel, dataset: &mut ThreadedDataloader<Vec<u32>, LoadingState<T>>, optimizer: &mut Optimizer<AdamW>) -> (f64, f64) {
    model.train();
    let bar = train_progress_bar((dataset.len() / BATCH_SIZE) as u64);
    let mut loss_avg = ExponentialAverage::new();
    let mut accuracies: Vec<f64> = vec![];
    
    for (i, batch) in dataset.enumerate() {
        optimizer.zero_grad();
        // Process batch into tensors
        let (mut inputs, mut targets) = collate_function(batch);
        let (batch_size, seq_length) = inputs.size2().unwrap();
        inputs = inputs.to_device(Device::cuda_if_available());
        targets = targets.to_device(Device::cuda_if_available());
        // Run through model
        let output = model.forward(&inputs);

        // Get loss
        let loss = output.view([batch_size * seq_length, output.size()[2]])
            .cross_entropy_for_logits(&targets.view([batch_size * seq_length]).to_kind(Kind::Int64));
        loss.backward();
        loss_avg.update(f64::from(loss).exp());
        if i % BATCH_AGGREGATIONS == 0 {
            optimizer.step();
            optimizer.zero_grad();
        }

        // Get accuracy
        accuracies.push(f64::from(output.accuracy_for_logits(&targets.to_kind(Kind::Int64))));

        bar.inc(1);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));
    }
    (loss_avg.value, accuracies.iter().sum::<f64>() / accuracies.len() as f64)
}

fn test_epoch<T: Tokenizer>(model: &mut LanguageModel, dataset: &mut ThreadedDataloader<Vec<u32>, LoadingState<T>>) -> (f64, f64) {
    model.eval();
    let mut losses = vec![];
    let mut accuracies: Vec<f64> = vec![];
    let bar = test_progress_bar((dataset.len() / BATCH_SIZE) as u64);
    for batch in dataset {
        // Process batch into tensors
        let (mut inputs, mut targets) = collate_function(batch);
        let (batch_size, seq_length) = inputs.size2().unwrap();
        inputs = inputs.to_device(Device::cuda_if_available());
        targets = targets.to_device(Device::cuda_if_available());
        // Run through model
        let output = model.forward(&inputs);

        // Get loss
        let loss = output.view([batch_size * seq_length, output.size()[2]])
            .cross_entropy_for_logits(&targets.view([batch_size * seq_length]).to_kind(Kind::Int64));
        losses.push(f64::from(loss).exp());

        // Get accuracy
        accuracies.push(f64::from(output.accuracy_for_logits(&targets.to_kind(Kind::Int64))));
        bar.inc(1);
    }
    (losses.iter().sum::<f64>() / losses.len() as f64, accuracies.iter().sum::<f64>() / accuracies.len() as f64)
}

fn generate<T: Tokenizer>(sentence: String, model: &mut LanguageModel, tokens_to_generate: usize, temperature: f64, local_vocab: &Vocab, tokenizer: &T) -> String {
    model.eval();
    let mut tokens = tokenizer.tokenize(&sentence);
    for _ in 0..tokens_to_generate {
        // Generate new token
        // Vectorize sentence
        let vectorized: Vec<i32> = local_vocab.indexes_from_tokens(&tokens).unwrap()
            .iter().map(|i| {*i as i32}).collect();
        let input = Tensor::of_slice(&vectorized).unsqueeze(0).to(Device::cuda_if_available());
        let output = model.forward(&input);
        let index = sample_1d(output.squeeze_dim(0).i(-1), temperature);
        let token = local_vocab.index2token[index].clone();
        tokens.push(token);
    }
    tokenizer.untokenize(tokens)
}

/// Basic language model training
#[derive(Clap, Debug)]
#[clap(name = "Transformer Language Model")]
struct Args {
    /// Whether or not to attempt to load the model before training
    #[clap(short, long)]
    load: bool,

    /// Whether or not to run testing instead of training
    #[clap(short, long)]
    test: bool
}